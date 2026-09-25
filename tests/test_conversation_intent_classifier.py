import pytest

from app.intent.classifier import ConversationIntentClassifier

classifier = ConversationIntentClassifier()


def _memory(messages: list[dict[str, str]] | None = None) -> dict:
    return {"messages": messages or []}


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        # Spec's own Draft Generation examples.
        ("Generate FIR draft.", "Draft Generation"),
        ("Draft a police complaint.", "Draft Generation"),
        ("Prepare RTI application", "Draft Generation"),
        ("Write legal notice.", "Draft Generation"),
        ("Create affidavit.", "Draft Generation"),
    ],
)
def test_classifies_clear_draft_requests(message: str, expected_intent: str) -> None:
    assert classifier.classify(message, _memory()).intent == expected_intent


@pytest.mark.parametrize(
    "message",
    [
        # Spec's own "should NOT trigger Draft Generation" examples.
        "What is FIR?",
        "Explain FIR.",
        "FIR ka process kya hai?",
        "Difference between FIR and NCR.",
        "What is Zero FIR?",
        # Pre-existing false positives fixed alongside this feature.
        "I need a lawyer",
        "Can you recommend a lawyer for cheque bounce",
        "I want to know about RTI",
    ],
)
def test_does_not_classify_informational_or_lawyer_asks_as_draft(message: str) -> None:
    assert classifier.classify(message, _memory()).intent != "Draft Generation"


def test_lawyer_recommendation_explicit_ask() -> None:
    for message in ["Can you recommend a lawyer for cheque bounce", "vakil chahiye for property dispute"]:
        match = classifier.classify(message, _memory())
        assert match.intent == "Lawyer Recommendation"


def test_translation_with_target_language_is_not_ambiguous() -> None:
    # `memory["messages"]` mirrors chat_service's real shape: the current message
    # is already appended as the last entry by the time classify() runs.
    memory = _memory(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "Translate this to Hindi"}]
    )
    match = classifier.classify("Translate this to Hindi", memory)
    assert match.intent == "Translation"
    assert not match.ambiguous
    assert match.resolved_translation_target == "hindi"


def test_translation_without_target_language_is_ambiguous() -> None:
    memory = _memory(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "Can you translate that for me?"}]
    )
    match = classifier.classify("Can you translate that for me?", memory)
    assert match.intent == "Translation"
    assert match.ambiguous
    assert match.resolved_translation_target is None


def test_translation_with_no_prior_reply_is_ambiguous() -> None:
    match = classifier.classify("Translate this to Hindi", _memory())
    assert match.intent == "Translation"
    assert match.ambiguous


@pytest.mark.parametrize("message", ["Summarize our conversation", "sankshep de do"])
def test_summarization(message: str) -> None:
    assert classifier.classify(message, _memory()).intent == "Summarization"


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("Meaning of habeas corpus", "Legal Dictionary"),
        ("Difference between FIR and NCR", "Law Comparison"),
        ("BNS vs IPC", "Law Comparison"),
        ("What is the process to file an FIR?", "Legal Procedure"),
        ("FIR kaise file karein", "Legal Procedure"),
        ("Review my rental agreement", "Document Analysis"),
        ("Explain Section 138", "Legal Explanation"),
        # Part 50: "explain"/"summarize" a named upload must route to
        # Document Analysis (summarize-and-flag-risks shape), not a generic
        # definition/summary hint -- "Explain FIR."/"Explain Section 138"
        # (no document noun) must still route past this unaffected.
        ("pdf explain kro", "Document Analysis"),
        ("explain the uploaded file", "Document Analysis"),
        ("Explain FIR.", "Legal Explanation"),
    ],
)
def test_same_routing_categories(message: str, expected_intent: str) -> None:
    assert classifier.classify(message, _memory()).intent == expected_intent


def test_followup_requires_a_prior_assistant_turn() -> None:
    with_history = _memory(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "What about that?"}]
    )
    assert classifier.classify("What about that?", with_history).intent == "Follow-up Question"

    first_turn = _memory([{"role": "user", "content": "What about that?"}])
    assert classifier.classify("What about that?", first_turn).intent != "Follow-up Question"


@pytest.mark.parametrize(
    "message,expected_general_conversation",
    [
        ("Hi, my landlord won't return my deposit", False),
        ("Namaste, mera FIR ke baare mein sawaal hai", False),
        ("Thanks, that helps", True),
        ("Namaste, kaise ho?", True),
        ("Hi", True),
        ("ok", True),
        # Part 41 regression: "hi" had no "+" repetition (unlike "hello+"/
        # "hey+" right next to it in `_GREETING_UNIT`), so the extremely
        # common casual "hii"/"hiii" fell through to the "Legal Dictionary"
        # bare-term-lookup fallback and reached the RAG/general-knowledge
        # pipeline for a message with zero legal content -- confirmed root
        # cause of a production regression where "hii" produced a
        # completely unrelated hallucinated answer.
        ("hii", True),
        ("hiii", True),
        ("hiiii", True),
        ("hello", True),
        # Must NOT become an unwanted false positive from the "hi+" fix.
        ("history", False),
        ("hindi kanoon kya hai", False),
        # Part 41 regression test item 9.
        ("aur kya haal hai", True),
        ("kya haal hai", True),
        # Must NOT become an unwanted false positive from the "kya haal"
        # addition -- a real FIR question must still route past it.
        ("fir kya hota hai", False),
        ("accha fir kya hota hai", False),
        # Part 48: "accha" ("oh, I see") and "theek hai"/"theek" ("okay"/
        # "fine") are two of the most common Hinglish casual acknowledgments
        # -- same class of gap as the "hii"/"kya haal" fixes above.
        ("accha", True),
        ("theek hai", True),
        ("theek", True),
        ("accha theek hai", True),
    ],
)
def test_general_conversation_greeting_strip_precision(message: str, expected_general_conversation: bool) -> None:
    intent = classifier.classify(message, _memory()).intent
    assert (intent == "General Conversation") is expected_general_conversation


def test_bare_what_is_question_is_never_draft_or_general_conversation() -> None:
    match = classifier.classify("What is bail?", _memory())
    assert match.intent not in {"Draft Generation", "General Conversation"}


def _memory_with_prior_reply(user_text: str) -> dict:
    return _memory([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": user_text}])


@pytest.mark.parametrize(
    "message,expected_modification_type",
    [
        ("Summarize.", "summarize"),
        ("TL;DR", "summarize"),
        ("30 words", "shorten"),
        ("Make it shorter", "shorten"),
        ("Explain in detail.", "expand"),
        ("Elaborate", "expand"),
        ("Explain simply.", "simplify"),
        ("ELI5", "simplify"),
        ("Bullet points.", "bullets"),
        ("Key points", "bullets"),
        ("Rewrite professionally.", "rewrite"),
        ("Professional.", "rewrite"),
        ("Table.", "table"),
        ("Compare.", "table"),
        ("Pros and cons", "table"),
        ("Give examples.", "examples"),
        ("Timeline.", "timeline"),
        ("Procedure.", "timeline"),
        ("Mind map.", "visual"),
    ],
)
def test_response_modification_commands_with_prior_reply(message: str, expected_modification_type: str) -> None:
    match = classifier.classify(message, _memory_with_prior_reply(message))
    assert match.intent == "Response Modification"
    assert not match.ambiguous
    assert match.modification_type == expected_modification_type


def test_response_modification_without_prior_reply_is_ambiguous() -> None:
    match = classifier.classify("Bullet points.", _memory())
    assert match.intent == "Response Modification"
    assert match.ambiguous


def test_response_modification_does_not_swallow_genuine_new_questions() -> None:
    # These carry real topical content beyond the bare command, so they must
    # still fall through to their normal routing even with a prior reply.
    memory = _memory_with_prior_reply("placeholder")
    assert classifier.classify("Compare FIR and NCR", memory).intent == "Law Comparison"
    assert classifier.classify("What is the process to file an FIR?", memory).intent == "Legal Procedure"
    assert classifier.classify("Explain Section 138", memory).intent == "Legal Explanation"


def test_simple_hindi_simplify_also_resolves_target_language() -> None:
    match = classifier.classify("Simple Hindi.", _memory_with_prior_reply("Simple Hindi."))
    assert match.intent == "Response Modification"
    assert match.modification_type == "simplify"
    assert match.resolved_translation_target == "hindi"


def test_bare_language_reply_resolves_pending_translation_clarification() -> None:
    memory = _memory_with_prior_reply("Hindi")
    memory["pending_clarification"] = "translation_target"
    match = classifier.classify("Hindi", memory)
    assert match.intent == "Translation"
    assert not match.ambiguous
    assert match.resolved_translation_target == "hindi"


def test_bare_language_reply_without_pending_clarification_is_still_translation() -> None:
    # Reversed from the original "is NOT Translation" design: a real
    # conversation trace ("What is FIR?" -> "30 words" -> "Hindi") showed
    # users skip straight to naming the language without saying "Translate."
    # first, and the previous behavior (falling through to a bare-term
    # "Legal Dictionary" lookup, i.e. trying to define the word "Hindi" as
    # a legal term) was a clear worse guess than assuming translation.
    match = classifier.classify("Hindi", _memory_with_prior_reply("Hindi"))
    assert match.intent == "Translation"
    assert match.resolved_translation_target == "hindi"


@pytest.mark.parametrize(
    "message",
    [
        "What did I ask?",
        "What were we discussing?",
        "What was my previous question?",
        "What did you just explain?",
        "Which topic were we talking about?",
        "Can you remind me?",
        "What was stolen?",
        "What did I upload?",
        "Which draft were we creating?",
        "Continue from before.",
        "Continue.",
        "What was my first question?",
        "What topic are we on?",
    ],
)
def test_conversation_memory_questions_with_prior_reply(message: str) -> None:
    match = classifier.classify(message, _memory_with_prior_reply(message))
    assert match.intent == "Conversation Memory"
    assert not match.ambiguous


@pytest.mark.parametrize("message", ["previous question kya tha?", "last question kya tha?", "first question kya tha?"])
def test_part49_hinglish_previous_question_recall(message: str) -> None:
    # Part 49: the English nouns "previous"/"question" code-switched into
    # Hindi sentence structure ("X kya tha?") were missing from the Hinglish
    # memory-recall pattern, which only covered the Hindi "pichla/pehla
    # sawaal" phrasing.
    match = classifier.classify(message, _memory_with_prior_reply(message))
    assert match.intent == "Conversation Memory"
    assert not match.ambiguous


def test_conversation_memory_without_prior_reply_is_ambiguous() -> None:
    match = classifier.classify("What did I ask?", _memory())
    assert match.intent == "Conversation Memory"
    assert match.ambiguous


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("What was the maximum bail amount?", "General Legal Information"),
        ("What was the outcome of Kesavananda Bharati case?", "General Legal Information"),
        ("Compare FIR and NCR", "Law Comparison"),
        ("What is the process to file an FIR?", "Legal Procedure"),
    ],
)
def test_conversation_memory_does_not_swallow_genuine_new_questions(message: str, expected_intent: str) -> None:
    match = classifier.classify(message, _memory_with_prior_reply(message))
    assert match.intent == expected_intent


# Part 42 "Answer Quality & Intent Accuracy": a short message that names its
# own legal topic must never be forced into "Follow-up Question" just
# because a prior (unrelated) assistant turn exists -- confirmed root cause
# of "Bail chahiye" after a Legal Notice discussion drifting into a Legal
# Notice answer instead of a bail one.
@pytest.mark.parametrize("message", ["Bail chahiye", "Bail chahiye.", "divorce chahiye"])
def test_standalone_legal_topic_is_not_forced_into_followup(message: str) -> None:
    memory = _memory(
        [
            {"role": "user", "content": "Legal notice kaise likhen?"},
            {"role": "assistant", "content": "Yahan legal notice likhne ka tarika hai..."},
            {"role": "user", "content": message},
        ]
    )
    match = classifier.classify(message, memory)
    assert match.intent != "Follow-up Question"


# Genuine pronoun-based continuations must still resolve as follow-ups --
# regression guard on the narrowing above not being too aggressive.
@pytest.mark.parametrize(
    "message",
    ["Uske baad kya karu?", "Documents kaunse chahiye?", "Kitna time lagta hai?", "What about that?"],
)
def test_genuine_followups_still_classified_as_followup(message: str) -> None:
    match = classifier.classify(
        message,
        _memory(
            [
                {"role": "user", "content": "Mera phone chori ho gaya."},
                {"role": "assistant", "content": "Sabse pehle FIR file karein..."},
                {"role": "user", "content": message},
            ]
        ),
    )
    assert match.intent == "Follow-up Question"


@pytest.mark.parametrize(
    "message",
    [
        "Aaj weather kaisa hai?",
        "Python me list aur tuple me kya difference hai?",
        "Mujhe ek horror story sunao.",
        "Tell me a joke.",
    ],
)
def test_off_domain_messages_never_enter_legal_routing(message: str) -> None:
    # First-turn, no prior context to inherit from either.
    assert classifier.classify(message, _memory()).intent == "Out of Domain"
    # With a prior unrelated legal turn -- must not inherit stale context.
    memory = _memory(
        [
            {"role": "user", "content": "Bail kaise milti hai?"},
            {"role": "assistant", "content": "Bail milne ka process yeh hai..."},
        ]
    )
    assert classifier.classify(message, memory).intent == "Out of Domain"


def test_non_indian_jurisdiction_legal_question_is_flagged() -> None:
    match = classifier.classify("US me divorce ka process kya hai?", _memory())
    assert match.intent == "Non-Indian Jurisdiction"


def test_indian_jurisdiction_divorce_question_is_not_flagged() -> None:
    match = classifier.classify("India me divorce ka process kya hai?", _memory())
    assert match.intent != "Non-Indian Jurisdiction"


@pytest.mark.parametrize("message", ["Hello", "Aur kya haal hai?"])
def test_greetings_still_general_conversation_not_off_domain(message: str) -> None:
    assert classifier.classify(message, _memory()).intent == "General Conversation"


# Part 47 "Answer Relevance & Correctness Pass" -- regression tests for the
# 15 named test cases. Cases 1/5/10 exercise the two classifier gaps fixed
# in this part (Hindi "kya hota/hoti hai" definition shape, and causative
# "kaise karwayein/milegi" procedural shape); the rest confirm behavior that
# was already correct once `messages` mirrors chat_service's real shape
# (current turn already appended) is genuinely locked in, not accidentally
# broken by the two regex changes above.


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        # 1: bare Hindi definition question -- previously fell through every
        # specific pattern into the generic 0.4-confidence catch-all instead
        # of landing on the same "definition first" shaping as its English
        # equivalent.
        ("FIR kya hota hai?", "Legal Explanation"),
        # 2: English equivalent, already worked (regression guard).
        ("What is FIR?", "Legal Explanation"),
        # 4: "why should I" -- no dedicated pattern for this shape exists;
        # it correctly lands in the generic informational catch-all, which
        # still carries the same shape-to-the-question hint.
        ("FIR kyun karwani chahiye?", "General Legal Information"),
        # 5: causative "how do I get an FIR done" -- previously misread as a
        # bare 3-token term and routed to "Legal Dictionary" (definition-
        # first hint) instead of "Legal Procedure" (step-by-step hint).
        ("FIR kaise karwayein?", "Legal Procedure"),
        # 6/mera bike scenario opener.
        ("mera bike chori ho gaya kya karu?", "Legal Advice"),
        # 10: same Hindi-definition gap as case 1, different topic word.
        ("bail kya hoti hai?", "Legal Explanation"),
        ("hello", "General Conversation"),
        ("aur kya haal hai", "General Conversation"),
    ],
)
def test_part47_first_turn_intent_alignment(message: str, expected_intent: str) -> None:
    assert classifier.classify(message, _memory()).intent == expected_intent


def test_part47_procedure_pattern_does_not_overmatch_generic_howto_questions() -> None:
    # Guard on the `_PROCEDURE_PATTERN` broadening above: causative Hindi
    # verb forms were added, but the generic bare "karta/karti/karte" was
    # deliberately left out because it collides with countless everyday
    # "how do you do X" questions that have nothing to do with legal
    # procedure -- confirm one such off-domain example is still routed to
    # "Out of Domain", not swallowed into "Legal Procedure".
    match = classifier.classify("Python kaise seekhte hain?", _memory())
    assert match.intent == "Out of Domain"


def test_part47_followup_chain_preserves_active_topic() -> None:
    # 6-9: a short Hindi follow-up chain after an active bike-theft/FIR
    # topic must keep resolving as "Follow-up Question" (which then gets
    # LLM-rewritten against the prior turns before retrieval, not RAG'd on
    # the isolated fragment) -- not misread as a bare-term lookup.
    def _memory_after_bike_theft(current_text: str) -> dict:
        return _memory(
            [
                {"role": "user", "content": "mera bike chori ho gaya kya karu?"},
                {"role": "assistant", "content": "Sabse pehle FIR file karein nazdeeki police station mein..."},
                {"role": "user", "content": current_text},
            ]
        )

    for message in ["aisa kyu?", "fir kya?", "uske baad?"]:
        match = classifier.classify(message, _memory_after_bike_theft(message))
        assert match.intent == "Follow-up Question", f"{message!r} -> {match.intent}"


def test_part49_extended_followup_chain_real_transcript() -> None:
    # Part 49 regression: the exact real-transcript follow-up chain after a
    # bike-theft opener, each turn built on the PRIOR turn's own reply (not
    # a fixed opener) since real conversations accumulate context turn by
    # turn. "aur agar police mana kar de?" (6 tokens) is the specific
    # confirmed failure -- it exceeded the generic short-message fallback's
    # 5-token cap and had no Hindi "and if" pattern to match instead.
    messages = [
        {"role": "user", "content": "Meri bike chori ho gayi hai, kya karu?"},
        {"role": "assistant", "content": "Sabse pehle FIR file karein nazdeeki police station mein..."},
    ]
    chain = [
        "uske baad kya karna hai?",
        "FIR kyu karwani chahiye?",
        "isme kitna time lagta hai?",
        "kaunse documents lagenge?",
        "aur agar police mana kar de?",
        "accha, fir?",
    ]
    for message in chain:
        match = classifier.classify(message, _memory(messages + [{"role": "user", "content": message}]))
        assert match.intent == "Follow-up Question", f"{message!r} -> {match.intent}"
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": "..."})


def test_part47_fact_recall_questions_are_not_general_conversation_or_offdomain() -> None:
    # 13/14: fact-identifying recall questions ("what was stolen", "who hit
    # me") are answered directly from `entity_memory` in `chat_service`
    # BEFORE this classifier's intent is used for routing (see
    # `ChatService._store_entity_fact_if_any`/entity-memory recall gate) --
    # this classifier has no dedicated intent for them and isn't expected
    # to. The only regression this guards is the two new patterns above not
    # accidentally hijacking these into "Legal Explanation"/"Legal
    # Procedure" (which would never reach the entity-memory answer at all,
    # since only "Follow-up Question" triggers the LLM rewrite step, and
    # entity-memory checks the raw question text independent of intent).
    memory = _memory(
        [
            {"role": "user", "content": "mera bike chori ho gaya"},
            {"role": "assistant", "content": "Sabse pehle FIR file karein..."},
            {"role": "user", "content": "mera kya chori hua tha?"},
        ]
    )
    match = classifier.classify("mera kya chori hua tha?", memory)
    assert match.intent not in {"Legal Explanation", "Legal Procedure"}


def test_part47_unrelated_question_after_active_topic_is_not_swallowed_as_followup() -> None:
    # 15: a genuinely new, unrelated legal question after an active topic
    # must get its own answer, not be forced into the follow-up thread.
    memory = _memory(
        [
            {"role": "user", "content": "mera bike chori ho gaya kya karu?"},
            {"role": "assistant", "content": "Sabse pehle FIR file karein..."},
            {"role": "user", "content": "cheque bounce hone par kya hota hai?"},
        ]
    )
    match = classifier.classify("cheque bounce hone par kya hota hai?", memory)
    assert match.intent != "Follow-up Question"


# Regressions from a live UI transcript (2026-09-21).
def test_off_topic_chitchat_is_out_of_domain_not_a_legal_question() -> None:
    assert classifier.classify("pizza khana hai", {}).intent == "Out of Domain"


@pytest.mark.parametrize("reply", ["kasmiri", "Devanagari", "Kashmiri", "hindi"])
def test_bare_language_reply_completes_a_pending_translation(reply: str) -> None:
    memory = _memory_with_prior_reply(reply)
    memory["pending_clarification"] = "translation_target"
    match = classifier.classify(reply, memory)
    assert match.intent == "Translation"
    assert match.resolved_translation_target in {"kashmiri", "hindi"}


@pytest.mark.parametrize(
    "message",
    [
        "Ab corrected deposit amount, disputed deduction, landlord ka naam aur keys handover date batao.",
        "Mera naam, landlord ka naam aur pending security deposit batao.",
    ],
)
def test_reading_back_facts_the_user_gave_is_conversation_memory(message: str) -> None:
    assert classifier.classify(message, _memory_with_prior_reply(message)).intent == "Conversation Memory"


def test_a_general_legal_amount_question_is_not_memory_recall() -> None:
    assert classifier.classify("cheque bounce ka penalty amount batao", _memory_with_prior_reply("x")).intent != "Conversation Memory"
