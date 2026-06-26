from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Iterable


def coerce_index_terms(value: Any) -> list[str]:
    """Convert user-facing ids, titles, and aliases into searchable strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            terms.extend(coerce_index_terms(item))
        return terms
    if isinstance(value, dict):
        terms = []
        for key in ("id", "key", "name", "title", "slug"):
            if key in value:
                terms.extend(coerce_index_terms(value[key]))
        if terms:
            return terms
        return [str(value)]

    text = str(value).strip()
    return [text] if text else []


@dataclass
class DocumentIndexBuilder:
    """Build a compact metadata index from document keys to contiguous row ranges."""

    index_name: str = "documents"
    entries: list[dict[str, Any]] = field(default_factory=list)
    _run_counts: dict[str, int] = field(default_factory=dict)

    def observe(
        self,
        row_id: int,
        key: Any,
        *,
        labels: Iterable[Any] = (),
        aliases: Iterable[Any] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        key_terms = coerce_index_terms(key)
        if not key_terms:
            return None

        canonical_key = key_terms[0]
        labels = coerce_index_terms(list(labels))
        aliases = coerce_index_terms([*key_terms[1:], *aliases])

        if self.entries:
            last = self.entries[-1]
            if (
                last.get("canonical_key") == canonical_key
                and int(last["row_start"]) + int(last["row_count"]) == int(row_id)
            ):
                last["row_count"] = int(last["row_count"]) + 1
                merge_unique(last.setdefault("labels", []), labels)
                merge_unique(last.setdefault("aliases", []), aliases)
                return last

        run_index = self._run_counts.get(canonical_key, 0) + 1
        self._run_counts[canonical_key] = run_index
        entry_key = canonical_key if run_index == 1 else f"{canonical_key}#run_{run_index}"
        if entry_key != canonical_key:
            aliases = [canonical_key, *aliases]

        entry: dict[str, Any] = {
            "key": entry_key,
            "canonical_key": canonical_key,
            "row_start": int(row_id),
            "row_count": 1,
            "aliases": unique_terms(aliases),
            "labels": unique_terms(labels),
        }
        if run_index != 1:
            entry["run_index"] = run_index
        if metadata:
            entry["metadata"] = dict(metadata)
        self.entries.append(entry)
        return entry

    def finish(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]

    def metadata_schema(
        self,
        *,
        key_column: str,
        label_columns: Iterable[str] = (),
        alias_columns: Iterable[str] = (),
    ) -> dict[str, Any]:
        return {
            "kind": "document_ranges",
            "index_name": self.index_name,
            "key_column": key_column,
            "label_columns": list(label_columns),
            "alias_columns": list(alias_columns),
            "entry_count": len(self.entries),
            "range_semantics": "Each entry points to a contiguous row range: [row_start, row_start + row_count).",
        }


def find_index_entries(entries: Iterable[dict[str, Any]], query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    entries = list(entries)
    if limit <= 0:
        return []
    query = str(query).strip()
    if not query:
        return entries[:limit]

    query_lower = query.lower()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, entry in enumerate(entries):
        terms = entry_terms(entry)
        lower_terms = [term.lower() for term in terms]
        score = 0.0
        if query_lower in lower_terms:
            score = 100.0
        elif any(term.startswith(query_lower) for term in lower_terms):
            score = 85.0
        elif any(query_lower in term for term in lower_terms):
            score = 70.0
        else:
            score = max((SequenceMatcher(None, query_lower, term).ratio() for term in lower_terms), default=0.0) * 60.0
        if score >= 38.0:
            scored.append((score, position, entry))

    scored.sort(key=lambda item: (-item[0], int(item[2].get("row_start", 0)), item[1]))
    return [entry for _score, _position, entry in scored[:limit]]


def index_entry_for_key(entries: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    entries = list(entries)
    key_lower = str(key).strip().lower()
    for entry in entries:
        if key_lower in [term.lower() for term in entry_terms(entry)]:
            return entry

    candidates = sorted({term for entry in entries for term in entry_terms(entry)})
    close = get_close_matches(str(key), candidates, n=5)
    suffix = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(f"Unknown RowPack index key {key!r}.{suffix}")


def entry_terms(entry: dict[str, Any]) -> list[str]:
    terms = coerce_index_terms(entry.get("key"))
    terms.extend(coerce_index_terms(entry.get("canonical_key")))
    terms.extend(coerce_index_terms(entry.get("aliases")))
    terms.extend(coerce_index_terms(entry.get("labels")))
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in ("title", "name", "author", "source", "id"):
            if key in metadata:
                terms.extend(coerce_index_terms(metadata[key]))
    return unique_terms(terms)


def unique_terms(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    merge_unique(out, values, seen=seen)
    return out


def merge_unique(target: list[str], values: Iterable[str], *, seen: set[str] | None = None) -> None:
    if seen is None:
        seen = {value.lower() for value in target}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        target.append(text)
        seen.add(key)
