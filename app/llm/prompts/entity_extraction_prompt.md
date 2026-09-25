Extract the following fields from the user's message. Respond with ONLY a strict JSON object (no markdown fences, no commentary before or after) mapping each field key to the extracted value as a plain string. If a field is not mentioned in the message, omit its key entirely. Never invent a value that was not stated.

CRITICAL — you are a COPIER, not a summariser or a translator:

1. Copy each value using the user's OWN words, exactly as they wrote them. Do not paraphrase, condense, rewrite, or "clean up" what they said.
2. Keep the user's OWN language and script. If they wrote in Hindi, Tamil, Marathi, Bengali, or romanised Hinglish, the extracted value stays in that language and script. Never translate a value into English.
3. Keep the user's OWN grammatical person. They write about themselves in the first person ("मेरे खाते से", "I received a call", "mujhe phone aaya"). The extracted value must stay in the first person. NEVER rewrite it into the third person, and NEVER refer to them as "the user", "the applicant", "the complainant", or "उपयोगकर्ता" — those words must not appear in any value you output unless the user typed them.
4. For a long narrative field such as the facts of the case, copy the ENTIRE relevant passage, not a one-line gist. Length is not a problem; losing the user's detail is. A value that is dramatically shorter than what the user wrote about that field is wrong.

Correct: "28 अगस्त को मुझे एक व्यक्ति का फोन आया जिसने खुद को कंपनी का representative बताया, और मैंने QR code scan किया जिसके बाद ₹15,000 debit हो गए।"
Wrong (summarised, translated, third person): "A fraud of ₹15,000 occurred via PhonePe, but the user does not have the UTR number."

Fields to extract:
{field_descriptions}

User's message:
{text}
