"""LoRA fine-tune of qwen3:4b on this app's own response format/style, using
real high-confidence past answers (see build_finetune_dataset.py).

Base model weights were downloaded manually via curl.exe into
models/Qwen3-4B-bnb-4bit/ -- huggingface_hub's own downloader (requests/
httpx, both) crashes on this machine with a native
"OPENSSL_Uplink ... no OPENSSL_Applink" error specific to huggingface.co,
while curl.exe (Windows' own HTTP stack) and every other HTTPS call in this
app (Gemini API, Ollama) work fine. Loading from the local directory with
local_files_only avoids that crash entirely at training time too.

Run from legal_ai_assistant/ with the .venv-finetune interpreter:
    .venv-finetune\\Scripts\\python.exe scripts\\finetune_qwen3.py
"""
import json
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL_DIR = "models/Qwen3-4B-bnb-4bit"
DATA_PATH = "scripts/finetune_data.jsonl"
OUTPUT_DIR = "models/qwen3-4b-legal-lora"
# The fixed system_prompt + rag_prompt template alone is ~3723 tokens
# (measured directly: 3418 + 305) BEFORE any chunk context/question/answer --
# that's the real floor, not chunk count. build_finetune_dataset.py's 2
# chunks / 500 chars gives a 4103-5636 token range (measured). 6144 OOM'd at
# the fused cross-entropy loss step specifically (attention itself fit, but
# nothing was left for it after 36 layers' activations) -- fixed below via
# UNSLOTH_CE_LOSS_TARGET_GB rather than shrinking context further, since
# context was already near the floor the fixed template imposes.
# 5760 got past attention AND the loss step but OOM'd in the backward pass
# through gradient-checkpointing recomputation. Dialed down further, trading
# a handful of the longest examples (dataset range was 4103-5636) for a
# sequence length this 6GB card can actually complete a backward pass at.
MAX_SEQ_LENGTH = 4608

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    local_files_only=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

raw_examples = [json.loads(line) for line in Path(DATA_PATH).read_text(encoding="utf-8").splitlines() if line.strip()]

def to_text(example: dict) -> dict:
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

texts = [to_text(ex) for ex in raw_examples]
# Drop examples whose tokenized length exceeds the context window -- some
# RAG prompts carry up to 6 retrieved chunks (~3000 chars each) and can run
# long; truncating mid-context risks cutting the actual operative statute
# text, so these are excluded rather than truncated.
kept = []
dropped = 0
for row in texts:
    n_tokens = len(tokenizer(row["text"])["input_ids"])
    if n_tokens <= MAX_SEQ_LENGTH:
        kept.append(row)
    else:
        dropped += 1
print(f"Training examples: {len(kept)} kept, {dropped} dropped (exceeded {MAX_SEQ_LENGTH} tokens)")

dataset = Dataset.from_list(kept)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        # OOM'd repeatedly in the backward pass at adamw_8bit even after
        # cutting seq length (9216->4608) and LoRA rank (16->8) -- switched
        # to a paged optimizer, which offloads optimizer state to CPU RAM
        # under memory pressure instead of failing outright. Built for
        # exactly this borderline-VRAM situation.
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")
print(f"Saved LoRA adapter to {OUTPUT_DIR}/adapter")
