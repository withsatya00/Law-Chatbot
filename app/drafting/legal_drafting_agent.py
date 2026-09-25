from __future__ import annotations

import json

import anthropic
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam
from pydantic import BaseModel, Field

MODEL = "claude-opus-5"


class DraftState(BaseModel):
    draft_type: str = ""
    extracted_facts: dict[str, str] = Field(default_factory=dict)
    missing_facts: list[str] = Field(default_factory=list)


class DraftResult(BaseModel):
    status: str  # "asking" | "ready"
    state: DraftState
    message: str | None = None
    draft_text: str | None = None


UPDATE_STATE_TOOL: ToolParam = {
    "name": "update_draft_state",
    "description": (
        "Record the draft type (if identifiable and not already known), any NEW facts extracted from "
        "the user's latest message, and the full list of facts still required but not yet known for "
        "this draft_type, based on what a real Indian legal professional would need to draft it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft_type": {"type": "string", "description": "E.g. 'Legal Notice', 'RTI Application', 'Police Complaint'. Omit if already known and unchanged."},
            "extracted_facts": {
                "type": "object",
                "description": "New or updated facts extracted from THIS message only, keyed by a short snake_case fact name.",
                "additionalProperties": {"type": "string"},
            },
            "missing_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Snake_case fact names still required for this draft_type that remain unknown.",
            },
        },
        "required": ["extracted_facts", "missing_facts"],
    },
}


class LegalDraftingAgent:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, model: str = MODEL) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model

    async def extract_information(self, user_input: str, current_state: DraftState) -> DraftState:
        system_prompt = (
            "You are a legal drafting intake assistant. Call `update_draft_state` with the draft_type "
            "(set/confirm if identifiable from context), any NEW facts extractable from THIS message "
            "(never repeat facts already known), and the complete list of facts still required but not "
            "yet known for this draft_type."
        )
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                MessageParam(
                    role="user",
                    content=(
                        f"Current draft_type: {current_state.draft_type or 'unknown'}\n"
                        f"Already known facts: {json.dumps(current_state.extracted_facts)}\n"
                        f"User's message: {user_input}"
                    ),
                )
            ],
            tools=[UPDATE_STATE_TOOL],
            tool_choice=ToolChoiceToolParam(type="tool", name="update_draft_state"),
            output_config={"effort": "medium"},
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        # `tool_use.input` is whatever the model produced -- the SDK types it
        # `object` precisely because the schema is advisory, not enforced. Each
        # field is checked against the shape `DraftState` declares before it is
        # used; anything else falls back to the state we already had, so a
        # malformed tool call loses that turn's extraction instead of raising a
        # pydantic error or, worse, writing a non-string into a draft field.
        data = tool_use.input if isinstance(tool_use.input, dict) else {}
        draft_type = data.get("draft_type")
        extracted = data.get("extracted_facts")
        missing = data.get("missing_facts")
        return DraftState(
            draft_type=draft_type if isinstance(draft_type, str) and draft_type else current_state.draft_type,
            extracted_facts={
                **current_state.extracted_facts,
                **(
                    {key: value for key, value in extracted.items() if isinstance(value, str)}
                    if isinstance(extracted, dict)
                    else {}
                ),
            },
            missing_facts=(
                [item for item in missing if isinstance(item, str)] if isinstance(missing, list) else []
            ),
        )

    async def process_draft_request(self, user_input: str, current_state: DraftState) -> DraftResult:
        updated_state = await self.extract_information(user_input, current_state)
        if updated_state.missing_facts:
            return DraftResult(status="asking", state=updated_state, message=self._followup_question(updated_state))
        draft_text = await self.generate_draft(updated_state)
        return DraftResult(status="ready", state=updated_state, draft_text=draft_text)

    def _followup_question(self, state: DraftState) -> str:
        facts = ", ".join(f.replace("_", " ") for f in state.missing_facts)
        return f"To draft this {state.draft_type or 'document'}, could you share: {facts}?"

    async def generate_draft(self, state: DraftState) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=(
                f"You are a legal drafting assistant. Generate a complete, properly formatted {state.draft_type} "
                "using ONLY the facts provided below -- never invent names, amounts, dates, or details not "
                "given. Use standard Indian legal drafting structure and formal register."
            ),
            messages=[{"role": "user", "content": f"Facts:\n{json.dumps(state.extracted_facts, indent=2)}"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
