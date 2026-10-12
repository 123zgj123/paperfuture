"""Canonical work identity: one paper, one id, whatever source named it.

A citation edge in this runtime is a string comparison. Both sides come from
different indexes that name the same paper differently, and nothing reconciled
them, so real citations did not match:

    an OpenAlex work        cite_id  = "doi:10.1104/pp.114.249"
    its own referenced_works entries = "W2041234567"

`_research_citation` strips a `doi:` prefix and folds arXiv versions. It has no
alias map, so two OpenAlex papers in a genuine citation relation could never
produce an edge. Semantic Scholar records a reference by arXiv id when it has
one and by DOI otherwise, while the citing work is keyed by whichever it has
first — so the two sides disagree there too.

This module gives every work one canonical id and remembers the aliases that
led to it. Two rules keep it from inventing lineage:

1. **An alias is evidence a source supplied**, never a guess. Titles resolve
   identity only when a source already tied them to the same external id.
2. **Merging two records never creates an edge.** It can only let an edge a
   source already reported find the work it points at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


ARXIV = "arxiv"
DOI = "doi"
S2 = "s2"
OPENALEX = "openalex"
NAMESPACES = (ARXIV, DOI, S2, OPENALEX)

_ARXIV_BARE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD = re.compile(r"^[a-z-]+(\.[A-Za-z]{2})?/\d{7}(v\d+)?$", re.I)
_OPENALEX_ID = re.compile(r"^[wW]\d+$")
_S2_ID = re.compile(r"^[0-9a-f]{40}$")


def _strip(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    return text


def parse_identifier(value: Any) -> tuple[str, str] | None:
    """Recognise one external identifier as (namespace, normalized value).

    Unrecognised text is not an identifier. A title is not an identifier here;
    it may resolve identity elsewhere, but it never names a work by itself.
    """
    text = _strip(value)
    if not text:
        return None
    lowered = text.lower()

    for prefix, namespace in (("arxiv.org/abs/", ARXIV), ("arxiv:", ARXIV),
                              ("doi.org/", DOI), ("doi:", DOI),
                              ("dx.doi.org/", DOI),
                              ("openalex.org/", OPENALEX), ("openalex:", OPENALEX),
                              ("api.openalex.org/works/", OPENALEX),
                              ("s2:", S2), ("semanticscholar.org/paper/", S2)):
        if lowered.startswith(prefix):
            rest = text[len(prefix):].strip().rstrip("/")
            if not rest:
                return None
            if namespace == ARXIV:
                return (ARXIV, re.sub(r"v\d+$", "", rest.lower()))
            if namespace == DOI:
                return (DOI, rest.lower())
            if namespace == OPENALEX:
                return (OPENALEX, rest.upper().rsplit("/", 1)[-1])
            return (S2, rest.lower())

    if _ARXIV_BARE.match(lowered) or _ARXIV_OLD.match(text):
        return (ARXIV, re.sub(r"v\d+$", "", lowered))
    if _OPENALEX_ID.match(text):
        return (OPENALEX, text.upper())
    if _S2_ID.match(lowered):
        return (S2, lowered)
    if lowered.startswith("10.") and "/" in lowered:
        return (DOI, lowered)
    return None


def normalize_title(value: str) -> str:
    """A comparison key for titles. Never an identifier on its own."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(value or ""))
    return " ".join(cleaned.split())


@dataclass
class CanonicalWork:
    """One paper, the ids that name it, and where each id came from."""

    canonical_id: str = ""
    arxiv_id: str = ""
    doi: str = ""
    s2_id: str = ""
    openalex_id: str = ""
    title: str = ""
    year: str = ""
    alias_sources: dict[str, str] = field(default_factory=dict)

    @property
    def identifiers(self) -> dict[str, str]:
        return {name: value for name, value in (
            (ARXIV, self.arxiv_id), (DOI, self.doi),
            (S2, self.s2_id), (OPENALEX, self.openalex_id)) if value}

    @property
    def keys(self) -> set[str]:
        """Every ``namespace:value`` this work answers to."""
        return {f"{name}:{value}" for name, value in self.identifiers.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id, "arxiv_id": self.arxiv_id,
            "doi": self.doi, "s2_id": self.s2_id, "openalex_id": self.openalex_id,
            "title": self.title, "year": self.year,
            "identifiers": self.identifiers,
            "alias_sources": dict(self.alias_sources),
        }

    def merge(self, other: "CanonicalWork") -> "CanonicalWork":
        """Absorb another record of the same work. Never invents an identifier."""
        for name in NAMESPACES:
            attr = {ARXIV: "arxiv_id", DOI: "doi", S2: "s2_id",
                    OPENALEX: "openalex_id"}[name]
            if not getattr(self, attr) and getattr(other, attr):
                setattr(self, attr, getattr(other, attr))
                if name in other.alias_sources:
                    self.alias_sources[name] = other.alias_sources[name]
        self.title = self.title or other.title
        self.year = self.year or other.year
        self.canonical_id = self._preferred()
        return self

    def _preferred(self) -> str:
        # A stable, human-checkable order. arXiv first because it is the id the
        # rest of this runtime already cites by.
        for name in (ARXIV, DOI, OPENALEX, S2):
            value = self.identifiers.get(name)
            if value:
                return f"{name}:{value}"
        return ""


def canonical_from_work(work: Any, *, source: str = "") -> CanonicalWork | None:
    """Build a canonical record from a FreshWork or a retrieved row."""
    row = work if isinstance(work, Mapping) else getattr(work, "to_dict", dict)()
    origin = str(source or row.get("source") or "")
    record = CanonicalWork(title=str(row.get("title") or ""),
                           year=str(row.get("published") or row.get("year") or "")[:4])
    candidates = [row.get("arxiv_id"), row.get("work_id"), row.get("cite_id"),
                  row.get("doi"), row.get("id"), row.get("url"),
                  *(row.get("aliases") or ())]
    ids = row.get("ids")
    if isinstance(ids, Mapping):
        candidates.extend(ids.values())
    for raw in candidates:
        parsed = parse_identifier(raw)
        if parsed is None:
            continue
        namespace, value = parsed
        attr = {ARXIV: "arxiv_id", DOI: "doi", S2: "s2_id", OPENALEX: "openalex_id"}[namespace]
        if not getattr(record, attr):
            setattr(record, attr, value)
            record.alias_sources[namespace] = origin or "supplied"
    record.canonical_id = record._preferred()
    return record if record.canonical_id else None


@dataclass
class WorkRegistry:
    """Resolve any identifier a source used to the one work it names.

    Identity is resolved by shared identifiers. A title match alone never joins
    two records: it only breaks a tie between records that already share one.
    """

    works: dict[str, CanonicalWork] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)

    def add(self, work: Any, *, source: str = "") -> CanonicalWork | None:
        record = (work if isinstance(work, CanonicalWork)
                  else canonical_from_work(work, source=source))
        if record is None:
            return None
        existing = None
        for key in record.keys:
            found = self.aliases.get(key)
            if found and found in self.works:
                existing = self.works[found]
                break
        if existing is None:
            title = normalize_title(record.title)
            candidate = self.titles.get(title) if title else None
            # A title only confirms a record that already shares an identifier.
            if candidate and candidate in self.works and (
                    self.works[candidate].keys & record.keys):
                existing = self.works[candidate]
        if existing is None:
            self.works[record.canonical_id] = record
            target = record
        else:
            previous = existing.canonical_id
            target = existing.merge(record)
            if target.canonical_id != previous:
                self.works.pop(previous, None)
                self.works[target.canonical_id] = target
                for key, value in list(self.aliases.items()):
                    if value == previous:
                        self.aliases[key] = target.canonical_id
        for key in target.keys:
            self.aliases[key] = target.canonical_id
        if target.title:
            self.titles.setdefault(normalize_title(target.title), target.canonical_id)
        return target

    def resolve(self, identifier: Any) -> str:
        """The canonical id this identifier names, or "" when unknown here."""
        parsed = parse_identifier(identifier)
        if parsed is None:
            return ""
        return self.aliases.get(f"{parsed[0]}:{parsed[1]}", "")

    def resolve_all(self, identifiers: Iterable[Any]) -> list[str]:
        found = [self.resolve(item) for item in identifiers or ()]
        return [item for item in dict.fromkeys(found) if item]

    def unresolved(self, identifiers: Iterable[Any]) -> list[str]:
        """Identifiers that name a work this registry has not seen.

        These are the citation targets worth fetching: an edge the source has
        already reported, pointing at a paper not yet in the graph.
        """
        out: list[str] = []
        for item in identifiers or ():
            parsed = parse_identifier(item)
            if parsed is None:
                continue
            key = f"{parsed[0]}:{parsed[1]}"
            if key not in self.aliases and key not in out:
                out.append(key)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "works": {key: row.to_dict() for key, row in sorted(self.works.items())},
            "alias_count": len(self.aliases),
            "note": ("Aliases are identifiers a source supplied. A title never "
                     "creates an identity on its own."),
        }
