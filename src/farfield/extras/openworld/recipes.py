"""Acquisition recipes: how a blocked research question may obtain a world.

`execute_acquire` used to support exactly one gap, `validator_snapshots`.
Everything else failed closed at `no_acquisition_recipe`, which made ACQUIRE the
most common terminal blockage in a run — while seventeen attested worlds sat on
disk that the controller had no way to bind. A research agent that cannot
obtain the object its question is about cannot answer the question, and no
amount of better scheduling changes that.

Two rules keep this from becoming a hole in the evidence boundary:

1. **A recipe never invents bytes.** It binds an already-attested world, or it
   derives a child from an attested parent. Both paths end at the existing
   attestation check; a world that cannot be attested is not created.
2. **No model-chosen fetching.** No recipe here reaches the network. If a
   network recipe is ever added it must carry an operator-pinned URL *and* an
   expected source digest, and refuse on mismatch. Letting a model name a URL
   would let it choose its own evidence.

An unknown gap still fails closed, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..world import load_fixture
from .worldio import is_attested


CATALOG_BINDING = "catalog_binding"
DERIVED = "derived"
CONSTRUCT_TEMPLATE = "construct_template"
RECIPE_KINDS = (CATALOG_BINDING, DERIVED, CONSTRUCT_TEMPLATE)


@dataclass(frozen=True)
class AcquisitionRecipe:
    """A declared, checkable way to satisfy one capability or observable gap."""

    kind: str
    gap: str
    world_id: str = ""
    schema: str = ""
    domains: tuple[str, ...] = ()
    root: str = ""
    digest: str = ""
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "gap": self.gap, "world_id": self.world_id,
            "schema": self.schema, "domains": list(self.domains), "root": self.root,
            "digest": self.digest, "why": self.why,
            "bytes_origin": (
                "already_attested_catalog_world" if self.kind == CATALOG_BINDING
                else "operator_owned_template" if self.kind == CONSTRUCT_TEMPLATE
                else "derived_from_attested_parent"
            ),
        }


def _gap_terms(gap: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(gap or ""))
    return {word for word in cleaned.split() if len(word) >= 3}


def catalog_worlds(catalog: Path | None) -> list[Any]:
    """Attested worlds available on disk. Unattested directories are skipped."""
    if catalog is None or not Path(catalog).is_dir():
        return []
    found = []
    for entry in sorted(Path(catalog).iterdir()):
        if not entry.is_dir() or not (entry / "manifest.json").is_file():
            continue
        try:
            fixture = load_fixture(entry)
        except (OSError, ValueError, KeyError):
            continue
        if is_attested(fixture):
            found.append(fixture)
    return found


def match_catalog_world(
    gap: str,
    catalog: Path | None,
    *,
    exclude: Iterable[str] = (),
) -> AcquisitionRecipe | None:
    """Find an attested world whose declared schema or domains provide the gap.

    Matching is on the world's own manifest — the schema and domains it was
    frozen with — not on a model's opinion that a dataset looks relevant. A gap
    naming nothing the catalog declares finds nothing, which is the correct
    answer rather than the nearest dataset.
    """
    terms = _gap_terms(gap)
    if not terms:
        return None
    skip = {str(item) for item in exclude}
    best: tuple[int, AcquisitionRecipe] | None = None
    for fixture in catalog_worlds(catalog):
        if fixture.id in skip:
            continue
        declared = _gap_terms(fixture.schema) | {
            str(item).lower() for item in (fixture.domains or ())}
        overlap = terms & declared
        if not overlap:
            continue
        recipe = AcquisitionRecipe(
            kind=CATALOG_BINDING, gap=str(gap), world_id=fixture.id,
            schema=fixture.schema, domains=tuple(fixture.domains or ()),
            root=str(fixture.root), digest=fixture.digest,
            why=f"attested catalog world declares {sorted(overlap)}")
        # Prefer the most specific declared overlap; ties break on world id so
        # the choice is reproducible.
        key = (len(overlap), fixture.id)
        if best is None or key > (best[0], best[1].world_id):
            best = (len(overlap), recipe)
    return best[1] if best else None


def resolve_recipe(
    gap: str,
    *,
    catalog: Path | None = None,
    parent_supports_derivation: bool = False,
    exclude: Iterable[str] = (),
    question: str = "",
) -> AcquisitionRecipe | None:
    """The recipe for this gap, or None. None means the run stays blocked.

    A bootstrapped run names its gap after whatever the seed world happened to
    offer — often the placeholder ``artifact_inventory`` — which describes
    nothing the catalog could ever declare. When the declared gap resolves to
    nothing, the research question itself is the only grounded statement of what
    object the work needs, so its terms are tried next.

    Matching a question's words against a world's declared domains proposes that
    this world may be the object of study. It does not claim the world answers
    the question: binding attested bytes makes measurement possible, and every
    claim still has to be registered and executed against them.
    """
    text = str(gap or "").strip()
    if text.startswith("construct_template:"):
        world_class = text.split(":", 1)[1]
        return AcquisitionRecipe(
            kind=CONSTRUCT_TEMPLATE, gap=text, schema=world_class,
            why="operator-owned template constructs a computational sandbox")
    if text == "validator_snapshots" and parent_supports_derivation:
        return AcquisitionRecipe(
            kind=DERIVED, gap=text, why="derive validator snapshots from the attested parent")
    if text:
        direct = match_catalog_world(text, catalog, exclude=exclude)
        if direct is not None:
            return direct
    asked = str(question or "").strip()
    if not asked:
        return None
    matched = match_catalog_world(asked, catalog, exclude=exclude)
    if matched is None:
        return None
    return AcquisitionRecipe(
        kind=matched.kind, gap=text or asked, world_id=matched.world_id,
        schema=matched.schema, domains=matched.domains, root=matched.root,
        digest=matched.digest,
        why=(f"{matched.why}; matched on the research question because the declared gap "
             f"{text or '(none)'!r} names nothing this catalog declares"))


def available_recipes(
    gaps: Iterable[str],
    *,
    catalog: Path | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Which of these gaps the runtime can currently satisfy, and how."""
    out: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        recipe = resolve_recipe(str(gap), catalog=catalog, exclude=exclude)
        if recipe is not None:
            out[str(gap)] = recipe.to_dict()
    return out
