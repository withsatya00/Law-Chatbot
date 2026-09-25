"""Draft template registry, loaded from `app/drafting/templates/*.yaml`.

Adding a new draft type is a pure data change: drop a new YAML file in this
directory (see `loader.py` for the required shape) — no Python code changes
needed anywhere in the drafting engine, API, or conversation flow.
"""

from app.drafting.templates.base import DraftTemplateDefinition
from app.drafting.templates.loader import get_template, list_templates

DRAFT_TEMPLATES: dict[str, DraftTemplateDefinition] = {
    template.draft_id: template for template in list_templates()
}

__all__ = ["DRAFT_TEMPLATES", "DraftTemplateDefinition", "get_template", "list_templates"]
