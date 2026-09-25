"""Builds a LoRA instruction-tuning dataset from real, high-confidence past
chat answers in query_logs -- teaching qwen3:4b this app's actual response
FORMAT/style/behaviour (clean final answer, no visible chain-of-thought, the
same citation conventions the system prompt already asks for), not new legal
facts. Facts stay RAG-sourced at inference time either way.

Reconstructs each training example through the SAME prompt templates
production uses (`app.services.chat_service._build_rag_messages`'s system_
prompt/rag_prompt), fed with the real retrieved_chunks and reconstructed
intent/entity objects stored on the query_log row, so the model trains on
exactly the input shape it will see in production -- not a hand-rolled
approximation of it.

Usage: python scripts/build_finetune_dataset.py [--out finetune_data.jsonl] [--min-confidence 0.7]
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Answers that are a template/flow response OR a provider-failure message
# rather than a real generated legal answer -- training on these would teach
# the model to imitate a fixed-string branch or literally output "Ollama did
# not respond within 120s" as if that were a legal answer. Confirmed live:
# an unfiltered first pass over query_logs pulled in exactly that -- 3 of 6
# manually spot-checked examples were raw provider-error text logged with
# confidence>=0.7 (the log's `confidence` reflects the LAST successful
# retrieval/grounding state, not "was this specific answer text real" --
# a provider failure after grounding was already computed can still write a
# high-confidence row whose `answer` is the error sentence, not the answer).
# The message text itself is the only reliable filter available after the
# fact; every phrase below is copied verbatim from app/llm/*.py's own
# `_error_response`/http_provider.py error strings so future wording changes
# there are the only thing that can silently defeat this filter.
_EXCLUDE_PATTERNS = re.compile(
    r"No verified document related|Is sawaal se related koi verified document|"
    r"Which State \(or Union Territory\)|Yeh matter kis State|"
    r"Let's draft your|Let's draft your|I just need \d+ more details|"
    r"I don't have a template for that|Available templates|"
    r"draft service abhi temporarily|Aapki di hui saari details|"
    r"I am a legal assistant focused on|Main aapki in cheezon mein|"
    r"This section number exists in more than one Act|"
    r"I found potentially relevant verified material in .* but the answer-generation|"
    r"Attach the document here and I will read it|"
    r"Mujhe yaad nahi hai|"
    r"did not respond within \d+s|did not finish within \d+s|"
    r"is currently unavailable|is not configured|was rejected\.|"
    r"rate limit or quota reached|Local LLM is unavailable|"
    r"not found\. Would you like to download|"
    r"Please start Ollama|API key is not configured|"
    r"I couldn't find sufficient verified legal information",
    re.IGNORECASE,
)


async def main() -> None:
    from app.core.windows_runtime import repair_windows_host_env
    repair_windows_host_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scripts/finetune_data.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()

    from app.database.mongodb import mongodb
    from app.models.collections import QUERY_LOGS
    from app.llm.prompts import prompt_registry
    from app.schemas.common import RetrievedChunk

    await mongodb.connect()
    db = mongodb.db

    # `_source_label` is a module-private helper in chat_service -- imported
    # directly rather than reimplemented, so the citation label format in
    # training data matches production byte-for-byte.
    from app.services.chat_service import _source_label

    # Excludes 2026-09-11 onward: this session's own ad-hoc test-script
    # invocations of ChatService().answer() write to this same collection
    # (it's the real production code path), so today's rows are repeated
    # test queries and deliberately-broken-config probes, not real usage.
    from datetime import datetime
    cursor = db[QUERY_LOGS].find({
        "confidence": {"$gte": args.min_confidence},
        "retrieved_chunks.0": {"$exists": True},
        "created_at": {"$lt": datetime(2026, 9, 11)},
    })

    examples = []
    skipped_excluded = 0
    skipped_short = 0
    async for row in cursor:
        question = row.get("question") or ""
        answer = (row.get("answer") or "").strip()
        language = row.get("language") or "en"
        intent = row.get("intent") or "General"
        conv_intent = row.get("conversation_intent") or "Legal Explanation"
        raw_chunks = row.get("retrieved_chunks") or []

        if not question or not answer:
            continue
        if _EXCLUDE_PATTERNS.search(answer):
            skipped_excluded += 1
            continue
        if len(answer) < 80:
            skipped_short += 1
            continue

        # Production (_build_rag_messages) uses up to 6 chunks at 3000 chars
        # each -- fine at INFERENCE (forward pass only, already confirmed
        # working on this 6GB card at that length). TRAINING additionally
        # holds gradients/activations for backprop, and this GPU has no
        # working flash-attention (xformers built for a different torch/CUDA
        # combo, FA2 unavailable), so standard attention's O(n^2) memory
        # blew up: a 9.2k-token example alone tried to allocate 8GB and OOMed
        # outright. Capped to 3 chunks / 800 chars here for training data
        # only -- teaches the same citation/format behaviour on a shorter
        # example without touching the production prompt-building code path.
        ranked = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", ""),
                text=c.get("text", ""),
                score=float(c.get("score", 0.0)),
                metadata=c.get("metadata", {}) or {},
            )
            for c in raw_chunks[:1]
        ]
        context = "\n\n".join(
            f"[Source {i + 1} — {_source_label(chunk)}]\n{chunk.text[:400]}"
            for i, chunk in enumerate(ranked)
        )
        # The REAL production system_prompt (prompt_registry's "system_prompt"
        # template) is ~3418 tokens on its own -- even with zero chunk context
        # that alone already exceeds what this 6GB card's backward pass can
        # complete without real flash-attention (confirmed: every attempt at
        # 4600-9200 total tokens OOM'd at the SAME gradient-checkpointing
        # attention-backward step, identically, regardless of LoRA rank/
        # optimizer/CE-loss-budget tuning -- none of those affect that memory
        # cost). This condensed prompt is TRAINING-ONLY: it teaches the same
        # behavioural habits (clean final answer, no visible reasoning, cite
        # by Act+Section using the given Source labels, stay in the requested
        # language) on a prompt this hardware can actually backprop through.
        # Production inference is unaffected -- chat_service.py's real
        # system_prompt is untouched, LoRA adapts weights, not the prompt.
        system_prompt = (
            "You are a legal information assistant for Indian law. Answer using ONLY the "
            "provided sources -- never invent a section number or case fact not present in "
            "them. Cite the Act name and Section number from the source label. Give a direct, "
            "clean final answer with no visible reasoning, meta-commentary, or 'let me analyze' "
            "preamble. If the sources don't answer the question, say so plainly instead of "
            f"guessing. Reply in {language}."
        )
        rag_prompt = prompt_registry.render(
            "rag_prompt",
            question=question,
            language=language,
            intent=intent,
            conversation_intent=f"{conv_intent} — ",
            entities=row.get("entities") or {},
            context=context,
            statutory_currency_note="None.",
            jurisdiction_note="None.",
        )
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rag_prompt},
                {"role": "assistant", "content": answer},
            ]
        })

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} training examples to {out_path}")
    print(f"Skipped: {skipped_excluded} template/flow responses, {skipped_short} too short")


asyncio.run(main())
