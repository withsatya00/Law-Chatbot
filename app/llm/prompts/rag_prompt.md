User question:
{question}

Detected language:
{language}

Intent:
{intent}

Conversation intent:
{conversation_intent}

Entities:
{entities}

Statutory currency note:
{statutory_currency_note}

Jurisdiction note:
{jurisdiction_note}

<context>
{context}
</context>

Answer the user's question directly and conversationally, grounded ONLY in the <context> above, per Rules 1-3 in the system prompt: if it genuinely answers this question, cite it and explain; if it's empty or actually about something else, output Rule 2's exact fallback line and nothing else. Shape your response around what this specific question needs — don't pad it with sections that don't apply here. Let the conversation intent's hint above (not just the legal-category intent) guide the shape of the answer. Weave in the relevant Act and section as part of the explanation rather than under separate headings, and only bring up practical next steps or required documents if the question is actually about going through a process (filing, drafting, disputing something), not a general legal concept. If the question is asking what a specific statutory provision itself says, follow the verbatim-quote-first rule above before explaining it. Cite sources by Act and section (or document name), never as "Source N" — those labels are internal to this prompt and meaningless to the reader. If the statutory currency note above is not "None.", follow it: lead with the provision currently in force and flag the old numbering as pre-July-2024. Write the whole reply in {language}.
