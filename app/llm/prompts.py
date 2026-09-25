from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def _load_prompt(prompt_dir: Path, name: str) -> str:
    """Reads one prompt template from disk, memoized by (directory, name).

    Deliberately a module-level function rather than an `@lru_cache` on the
    method: an `lru_cache` on a method keys on `self`, so the cache holds a
    strong reference to every `PromptRegistry` ever constructed for the life of
    the process. That is a leak (ruff B019) and it also makes the cache
    per-instance, so two registries pointing at the SAME directory each pay
    their own disk reads. Keying on the directory path instead shares the cache
    across instances and keeps no registry alive.
    """
    path = prompt_dir / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    return path.read_text(encoding="utf-8")


class PromptRegistry:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or Path(__file__).parent / "prompts"

    def load(self, name: str) -> str:
        return _load_prompt(self.prompt_dir, name)

    def render(self, name: str, **values: object) -> str:
        template = self.load(name)
        return template.format(**{key: str(value) for key, value in values.items()})


prompt_registry = PromptRegistry()
