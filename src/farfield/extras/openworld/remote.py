"""Remote conceptual search: where a research lineage leaves its local basin.

Principle 1 — remote search discovers a research basin; it does not define
every subsequent move.

Three moves may ask this module for a jump, and no other: REMOTE_BOOTSTRAP
(the first complete idea of the project), REMOTE_ESCAPE (a lineage the
discovery runtime has judged stuck) and the remote kind of
BRANCH_DIVERSIFICATION. Idea evolution never calls it. `discovery.plan_moves`
decides which move runs, and every idea artifact records whether a search ran,
so "the second hop did not jump" is a fact in the event log rather than a
promise in a docstring.

The legacy mission pipeline jumped on every hop and grew its distance band with
the hop index (`researchspace.band_for_index`), so later hops were chosen for
being far rather than for answering the previous hop's failure. The research
runtime then lost remote search altogether. This module restores it where it
belongs — at bootstrap and at the explicit escape/diversification moves above.

The space is the repository's deterministic local embedding (hashed TF-IDF with
a seeded projection, `embed.py`) over a concept corpus. It is a lexical space,
and a distance in it is provenance about where a frame came from. It is never a
measure of an idea's quality, novelty or progress.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PRODUCTION_CORPUS = "ds-arxiv-concepts-2026"
MIXED_CORPUS = "cs-mixed-arxiv-concepts-2026"
SHIPPED_CORPUS = "attn-concepts-s1"
REGISTRY = "concepts/arxiv_registry.json"

# Quantile band of distances from the local basin that a frame is drawn from.
# Not the farthest nodes: the tail of a lexical space is text the corpus has no
# words for, which is noise rather than a research area.
DEFAULT_BAND = (0.55, 0.9)
ANCHORS = 6


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_production_corpus(root: Path | None = None, requested: str = "") -> str:
    """Default corpus: mixed harvest when it exists, else the DS snapshot.

    Moved here from the legacy mission module so `farfield research` does not
    import a 6,700-line pipeline it never runs for one fifteen-line helper.
    """
    if str(requested or "").strip():
        return str(requested).strip()
    base = root or repository_root()
    mixed = base / "concepts" / MIXED_CORPUS / "G_full.json"
    if mixed.is_file():
        return MIXED_CORPUS
    return PRODUCTION_CORPUS


@dataclass(frozen=True)
class RemoteFrame:
    """One external conceptual frame an idea may leave its basin through."""

    label: str
    node_id: str
    distance: float
    degree: int = 0
    mechanism: str = ""
    assumptions: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "node_id": self.node_id,
                "distance": round(float(self.distance), 6), "degree": int(self.degree),
                "mechanism": self.mechanism, "assumptions": list(self.assumptions),
                "failure_conditions": list(self.failure_conditions), "source_ids": list(self.source_ids)}


@dataclass(frozen=True)
class RemoteSearchResult:
    purpose: str
    backend: str
    corpus: str = ""
    space_digest: str = ""
    local_basin: tuple[tuple[str, float], ...] = ()
    frames: tuple[RemoteFrame, ...] = ()
    band: tuple[float, float] = DEFAULT_BAND
    gap: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return as_remote_result({
            **dict(self.extra),
            "purpose": self.purpose,
            "backend": self.backend,
            "corpus": self.corpus,
            "space_digest": self.space_digest,
            "local_basin": [{"label": label, "distance": round(float(d), 6)}
                            for label, d in self.local_basin],
            "frames": [frame.to_dict() for frame in self.frames],
            "band": [float(self.band[0]), float(self.band[1])],
            "gap": self.gap,
            "distance_is_provenance_not_quality": True,
        }, purpose=self.purpose)


RemoteSearch = Callable[..., Any]


def as_remote_result(raw: Any, *, purpose: str) -> dict[str, Any]:
    """Record retrieval separately from attempted search and scientific usefulness.

    ``used`` means usable frames were returned for inclusion in the proposal
    context. It never means the proposal adopted a frame or gained from it.
    Backend flags cannot turn a failed or empty search into a successful jump.
    """
    from .inquiry import strings, text

    if isinstance(raw, RemoteSearchResult):
        return raw.to_dict()
    row = dict(raw or {}) if isinstance(raw, Mapping) else {}
    frames, rejected, seen = [], 0, set()
    items = row.get("frames") or []
    if not isinstance(items, (list, tuple)):
        items, rejected = [], 1
    for item in items:
        if isinstance(item, Mapping) and str(item.get("label") or "").strip():
            try:
                distance = float(item.get("distance") or 0.0)
                degree = int(item.get("degree") or 0)
            except (TypeError, ValueError, OverflowError):
                rejected += 1
                continue
            if not math.isfinite(distance) or distance < 0 or degree < 0:
                rejected += 1
                continue
            frame = {"label": str(item["label"]).strip(),
                     "node_id": str(item.get("node_id") or ""),
                     "distance": distance, "degree": degree}
            frame.update(mechanism=text(item.get("mechanism")),
                         assumptions=strings(item.get("assumptions")),
                         failure_conditions=strings(item.get("failure_conditions")),
                         source_ids=strings(item.get("source_ids")))
        elif isinstance(item, str) and item.strip():
            frame = {"label": item.strip(), "node_id": "", "distance": 0.0, "degree": 0}
        else:
            rejected += 1
            continue
        frame.setdefault("mechanism", "")
        for key in ("assumptions", "failure_conditions", "source_ids"):
            frame.setdefault(key, [])
        frame["epistemic"] = "GENERATED"
        frame["mechanism_status"] = ("reported_requires_source_check" if frame["mechanism"]
                                     else "concept_only_requires_landing_research")
        identity = frame["label"].casefold()
        if identity not in seen:
            frames.append(frame)
            seen.add(identity)
    backend = str(row.get("backend") or "supplied")
    if backend == "error":
        frames = []
    gap = str(row.get("gap") or ("" if frames else "no_remote_frame_returned"))
    if backend == "error" and not row.get("gap"):
        gap = "remote_search_failed"
    if not isinstance(raw, Mapping):
        gap = "remote_result_not_an_object"
    if rejected and not frames:
        gap = gap if row.get("gap") else "remote_frames_invalid"
    basin = row.get("local_basin") or []
    band = row.get("band") or DEFAULT_BAND
    return {
        "attempted": True,
        "used": bool(frames),
        "status": ("frames_returned" if frames else
                   "no_candidates" if gap in {"no_remote_frame_returned", "remote_band_empty",
                                               "no_local_basin_for_intent"} else "failed"),
        "rejected_frames": rejected,
        "purpose": purpose,
        "backend": backend,
        "corpus": str(row.get("corpus") or ""),
        "space_digest": str(row.get("space_digest") or ""),
        "requested_corpus": str(row.get("requested_corpus") or ""),
        "corpus_fallback": bool(row.get("corpus_fallback")),
        "retrieval_scope": str(row.get("retrieval_scope") or "unspecified"),
        "local_basin": list(basin) if isinstance(basin, (list, tuple)) else [],
        "frames": frames,
        "band": list(band) if isinstance(band, (list, tuple)) else list(DEFAULT_BAND),
        "gap": gap,
        "distance_is_provenance_not_quality": True,
    }


class ConceptSpaceSearch:
    """Remote frames from a concept corpus embedded in the local lexical space.

    The local basin is the concepts nearest the research intent
    (`farspace.select_near_anchors`). Frames are drawn from a middle-far band of
    distances from the basin centroid, excluding concepts whose terms are
    entirely contained in the intent, spaced evenly through the band and tie-broken by graph
    degree, so a frame is a developed concept rather than a stray title.
    Deterministic: the same corpus and intent give the same frames.
    """

    def __init__(self, corpus_id: str = "", *, root: Path | None = None,
                 graph_path: Path | None = None, frames: int = 3,
                 band: tuple[float, float] = DEFAULT_BAND) -> None:
        self.corpus_id = str(corpus_id or "")
        self.root = Path(root) if root is not None else repository_root()
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.frames = max(1, int(frames))
        self.band = (float(band[0]), float(band[1]))
        if not 0 <= self.band[0] < self.band[1] <= 1:
            raise ValueError("remote distance quantiles must satisfy 0 <= low < high <= 1")
        self._loaded: tuple[Any, Any, dict[str, str], str] | None = None

    def _resolve_graph(self) -> tuple[Path, str]:
        if self.graph_path is not None:
            return self.graph_path, self.corpus_id or self.graph_path.parent.name
        registry = json.loads((self.root / REGISTRY).read_text(encoding="utf-8"))
        wanted = resolve_production_corpus(self.root, self.corpus_id)
        by_id = {str(row.get("snapshot_id")): row for row in registry.get("snapshots") or []}
        # An explicit selection is a contract, not permission to substitute a corpus.
        candidates = (wanted,) if self.corpus_id else (wanted, SHIPPED_CORPUS)
        for corpus in candidates:
            row = by_id.get(corpus)
            if row and (self.root / str(row.get("full_graph") or "")).is_file():
                return self.root / str(row["full_graph"]), corpus
        raise FileNotFoundError(f"no concept graph is available for requested corpus {wanted}")

    def _load(self) -> tuple[Any, Any, dict[str, str], str]:
        if self._loaded is None:
            from ...graph import load_graph
            from ..embed import embed_nodes

            path, corpus = self._resolve_graph()
            graph = load_graph(path)
            space = embed_nodes(graph.nodes)
            label_of = {nid: str(node.get("title") or nid) for nid, node in graph.nodes.items()}
            self._loaded = (graph, space, label_of, corpus)
        return self._loaded

    def __call__(self, *, purpose: str, intent: str, context: Sequence[str] = (),
                 exclude: Sequence[str] = (), k: int | None = None) -> RemoteSearchResult:
        from ..domain import folded_terms
        from ..embed import EmbedError, cosine_distance
        from ..farspace import select_near_anchors

        wanted = max(1, int(k or self.frames))
        requested = self.corpus_id or ("" if self.graph_path is not None else
                                       resolve_production_corpus(self.root))
        provenance = {"requested_corpus": requested,
                      "retrieval_scope": "local_lexical_concepts_not_validated_structural_transfer"}
        try:
            graph, space, label_of, corpus = self._load()
            anchors = select_near_anchors(space, intent, label_of, k=ANCHORS)
        except (OSError, ValueError, KeyError, EmbedError) as exc:
            return RemoteSearchResult(purpose=purpose, backend="concept_space",
                                      gap=f"remote_space_unavailable:{exc}", extra=provenance)
        provenance["corpus_fallback"] = bool(requested and corpus != requested)
        if not anchors:
            return RemoteSearchResult(purpose=purpose, backend="concept_space", corpus=corpus,
                                      space_digest=space.digest, gap="no_local_basin_for_intent",
                                      extra=provenance)
        vectors = [space.vectors[nid] for _, nid in anchors if nid in space.vectors]
        centroid = [sum(values) / len(vectors) for values in zip(*vectors)]
        norm = math.sqrt(sum(v * v for v in centroid)) or 1.0
        centroid = tuple(v / norm for v in centroid)
        basin_terms = set(folded_terms(intent))
        for text in context:
            basin_terms |= set(folded_terms(str(text)))
        for _, nid in anchors:
            basin_terms |= set(folded_terms(label_of.get(nid, nid)))
        excluded = {str(label).strip().lower() for label in exclude}
        anchor_ids = {nid for _, nid in anchors}
        ranked = sorted((cosine_distance(centroid, vector), nid)
                        for nid, vector in space.vectors.items())
        lo = int(len(ranked) * self.band[0])
        hi = max(lo + 1, int(len(ranked) * self.band[1]))
        degree = {nid: len(graph.outgoing.get(nid, ())) + len(graph.incoming.get(nid, ()))
                  for _, nid in ranked[lo:hi]}
        pool = [
            (dist, nid) for dist, nid in ranked[lo:hi]
            if nid not in anchor_ids
            and label_of.get(nid, nid).strip().lower() not in excluded
            # Shared vocabulary can be the bridge to a transferable mechanism.
            # Only discard concepts wholly contained in the local vocabulary.
            and not (set(folded_terms(label_of.get(nid, nid))) <= basin_terms)
        ]
        pool.sort(key=lambda row: (round(row[0], 3), -degree.get(row[1], 0), row[1]))
        if not pool:
            return RemoteSearchResult(purpose=purpose, backend="concept_space", corpus=corpus,
                                      space_digest=space.digest, gap="remote_band_empty",
                                      local_basin=tuple((label_of.get(n, n), d) for d, n in anchors),
                                      extra=provenance)
        if wanted == 1 or len(pool) <= wanted:
            picks = pool[:wanted]
        else:
            step = (len(pool) - 1) / (wanted - 1)
            picks = [pool[round(i * step)] for i in range(wanted)]
        return RemoteSearchResult(
            purpose=purpose, backend="concept_space", corpus=corpus, space_digest=space.digest,
            local_basin=tuple((label_of.get(nid, nid), dist) for dist, nid in anchors),
            frames=tuple(RemoteFrame(label=label_of.get(nid, nid), node_id=nid, distance=dist,
                                     degree=degree.get(nid, 0)) for dist, nid in picks),
            band=self.band,
            extra=provenance,
        )
