Analyze the uploaded legal document using only the supplied document text. Do not invent facts,
parties, dates, clauses, legal authorities, or missing information. The legal context is optional
and must not be treated as facts about this document.

Document type: {document_type}
Response language: {language}
Document text:
{document_text}

Legal context:
{context}

Return ONLY valid JSON, with exactly this top-level shape:
{{
	"executive_summary": "brief factual summary",
	"legal_summary": "plain-language legal meaning and limitations",
	"important_clauses": ["short clause descriptions"],
	"analyzed_clauses": [
		{{"name": "clause name", "text": "short exact or faithful excerpt", "explanation": "what it means",
			 "risk_level": "low|medium|high", "risk_reason": "why this risk level applies"}}
	],
	"important_dates": ["date and its context"],
	"timeline": [{{"date": "ISO date or original date", "event_description": "factual event", "source_text": "supporting excerpt"}}],
	"important_names": ["party or person names"],
	"important_sections": ["section/article/clause references"],
	"key_risks": ["specific risk with its reason"],
	"action_items": ["specific review or follow-up action"],
	"missing_information": ["information genuinely needed but absent"],
	"structured_data": {{"document_type": "...", "parties": [], "obligations": []}},
	"confidence": 0.0
}}

Use empty arrays when the document does not support a value. Missing information must be
document-specific, not a generic disclaimer. Keep excerpts concise. Set confidence between 0 and 1.
