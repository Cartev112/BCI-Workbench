from __future__ import annotations

from typing import Protocol

from bciworkbench.ontology.packets import SignalPacket


class SignalSource(Protocol):
    """Source adapter contract for offline milestone sources."""

    def read(self) -> SignalPacket:
        ...


class OptionalDependencyError(ImportError):
    """Raised when an optional source dependency is unavailable."""


def require_optional(module_name: str, extra_name: str):
    try:
        return __import__(module_name)
    except Exception as exc:
        raise OptionalDependencyError(
            f"{module_name} support requires bciworkbench[{extra_name}]. "
            f'Install with: pip install "bciworkbench[{extra_name}]"'
        ) from exc

