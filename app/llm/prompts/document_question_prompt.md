Answer the question below using ONLY the document text provided. Do not invent, assume, or bring in outside legal knowledge, and do not answer from what a document of this type usually contains -- if the document text does not actually contain the answer, say so plainly rather than guessing.

Question: {question}
Response language: {language}

Document text (each page marked where known):
{document_text}

Return ONLY valid JSON, with exactly this shape:
{{
	"found_in_document": true or false,
	"answer": "a direct answer to the question, grounded only in the document text above -- empty string if not found",
	"supporting_quote": "a short, exact excerpt (under 40 words), copied verbatim from the document text, that supports the answer -- empty string if not found"
}}

If the question cannot be answered from the document text above, set "found_in_document" to false and leave "answer" and "supporting_quote" as empty strings -- never fabricate an answer and never fall back to general legal knowledge about documents of this type.
