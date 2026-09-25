"""The verified-lawyer-directory boundary.

There is no directory. That is the point of this module: it defines the
interface a real one would implement, and ships a provider that returns
nothing, so every caller is written against "there may be no listings" from
the start rather than having that discovered later.

Why this matters more than the usual "pluggable provider" story: a user
asking "which lawyer should I go to" is at their most vulnerable, and a
fabricated name with a fabricated phone number is not a harmless
placeholder — it is a person who does not exist, whom somebody may try to
call about a real legal problem, and possibly pay. So the rule is absolute:
**a lawyer's name, phone number, rating, availability or bar registration may
only ever be reported if a configured, verified directory returned it.**
Nothing in this codebase generates any of those.

`NullLawyerDirectory` is the default and, today, the only implementation.
`available()` returning `False` is what the recommendation engine and the
chat workflow both branch on, and both are tested against that branch.
"""

from typing import Protocol, runtime_checkable

from app.schemas.common import LawyerDirectoryListing

# Backward-compatible public name for provider implementations already built
# against the initial interface draft.
DirectoryListing = LawyerDirectoryListing


@runtime_checkable
class LawyerDirectory(Protocol):
    """What a real directory integration must provide."""

    def available(self) -> bool:
        """Whether this provider can return verified listings right now.

        A provider that is configured but unreachable must return `False`
        rather than raising: the recommendation is still useful without
        listings, and an outage must not take the answer down with it.
        """
        ...

    async def search(
        self, *, specialization: str, city: str = "", language: str = "", limit: int = 5
    ) -> list[DirectoryListing]:
        """Verified listings, or `[]`. Never a partial or inferred result."""
        ...


class NullLawyerDirectory:
    """The default: no directory is configured, so there are no listings.

    Deliberately not a stub that returns plausible sample data. Sample data
    in this position is indistinguishable from real data to the person
    reading it, and this is the one place in the product where that
    confusion could send somebody to a lawyer who does not exist.
    """

    name = "none"

    def available(self) -> bool:
        return False

    async def search(
        self, *, specialization: str, city: str = "", language: str = "", limit: int = 5
    ) -> list[DirectoryListing]:
        return []


_provider: LawyerDirectory = NullLawyerDirectory()


def get_directory() -> LawyerDirectory:
    return _provider


def set_directory(provider: LawyerDirectory) -> None:
    """Installs a real directory. Used by deployment wiring and by tests.

    Kept explicit rather than read from settings on every call so that
    "is a verified directory configured?" has one answer per process that a
    reader can trace, instead of depending on when an env var was read.
    """
    global _provider
    _provider = provider
