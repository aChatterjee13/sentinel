"""Source-tracing for merged Sentinel configs.

When ``extends:`` chains stack three or four files together, a Pydantic
validation error like ``alerts.channels.0.webhook_url: field required``
is unhelpful — the developer still has to chase the field through every
parent file by eye. ``ConfigSource`` records *which file* each leaf
value came from so the loader can append the originating path to error
messages.

The mechanism is intentionally simple:

1. Before merging two raw dicts, walk the child dict and tag every leaf
   value with a :class:`ConfigSource`.
2. Use the existing :func:`merge_dicts` semantics for the actual merge.
3. After merging, walk the merged dict, strip the tags into a
   :class:`SourceMap`, and pass the cleaned dict to Pydantic.
4. When validation fails, look up the offending JSON path in the
   :class:`SourceMap` to enrich the error message.

The map is kept side-by-side on the loader so it never pollutes the
schema or the validated :class:`SentinelConfig` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigSource:
    """Where a single config value originated from.

    Attributes:
        file: The YAML/JSON file that contributed this value.
        merged_from: When the same path is set in multiple parents, this
            holds the full chain (oldest → newest), in load order. The
            last entry always equals ``file``.
    """

    file: Path
    merged_from: tuple[Path, ...] = field(default_factory=tuple)

    def display(self) -> str:
        """Render a short, single-line summary for error messages."""
        if len(self.merged_from) <= 1:
            return str(self.file)
        chain = " → ".join(p.name for p in self.merged_from)
        return f"{self.file} (via {chain})"


class SourceMap:
    """A flat map from JSON paths (``alerts.channels.0.webhook_url``) to sources.

    The loader builds one of these per ``ConfigLoader.load()`` call.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ConfigSource] = {}

    def set(self, path: str, source: ConfigSource) -> None:
        self._entries[path] = source

    def get(self, path: str) -> ConfigSource | None:
        return self._entries.get(path)

    def lookup(self, loc: tuple[str | int, ...]) -> ConfigSource | None:
        """Find the closest source for a Pydantic ``loc`` tuple.

        Pydantic emits ``loc`` as ``("alerts", "channels", 0, "webhook_url")``.
        Convert it to a dotted path and walk back up if there's no exact
        match — the parent path's source is the next best thing.
        """
        parts = [str(p) for p in loc]
        while parts:
            path = ".".join(parts)
            entry = self._entries.get(path)
            if entry is not None:
                return entry
            parts.pop()
        return None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, path: str) -> bool:
        return path in self._entries

    def items(self) -> list[tuple[str, ConfigSource]]:
        return list(self._entries.items())


def annotate(data: dict[str, Any], source_file: Path) -> dict[str, Any]:
    """Wrap every leaf value in ``data`` with a ``(value, ConfigSource)`` marker.

    The marker is a 2-element tuple ``(value, ConfigSource)``. The merge
    helper preserves these tuples; :func:`harvest` strips them later.
    Lists are preserved structurally — each element is annotated.
    """
    src = ConfigSource(file=source_file, merged_from=(source_file,))

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return (value, src)

    result: dict[str, Any] = _walk(data)
    return result


def merge_with_sources(
    base: dict[str, Any],
    override: dict[str, Any],
    override_source: Path,
) -> dict[str, Any]:
    """Merge two annotated dicts, propagating source information.

    Override values replace base values; the override source becomes the
    "current" source for that field while the base source(s) live on in
    the ``merged_from`` chain.
    """
    result: dict[str, Any] = dict(base)
    for k, v in override.items():
        existing = result.get(k)
        # Recurse into nested dicts.
        if isinstance(existing, dict) and isinstance(v, dict):
            result[k] = merge_with_sources(existing, v, override_source)
            continue
        # Override leaf — bump the source chain.
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], ConfigSource):
            value, src = v
            chain: tuple[Path, ...]
            if (
                isinstance(existing, tuple)
                and len(existing) == 2
                and isinstance(existing[1], ConfigSource)
            ):
                chain = (*existing[1].merged_from, override_source)
            else:
                chain = src.merged_from
            result[k] = (value, ConfigSource(file=override_source, merged_from=chain))
        else:
            result[k] = v
    return result


def harvest(annotated: dict[str, Any]) -> tuple[dict[str, Any], SourceMap]:
    """Strip ``(value, ConfigSource)`` markers, returning the clean dict + map."""
    source_map = SourceMap()

    def _walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, _join(path, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v, _join(path, str(i))) for i, v in enumerate(value)]
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], ConfigSource):
            inner, src = value
            source_map.set(path, src)
            return inner
        return value

    cleaned = _walk(annotated, "")
    return cleaned, source_map


def _join(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"
