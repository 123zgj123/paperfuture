"""Bounded citation-graph expansion, and honest names for its edges.

The live path built transitions by looking for citations *inside one keyword
search's results*: six papers, pairwise, edge only when one happened to cite
another and the two indexes happened to name it the same way. In six live
segments that produced zero edges, so the trajectory lane never had anything to
retrieve.

Keyword search is a seed finder. A citation graph is built by dereferencing the
references a source already reported, and by asking which later works cite a
seed. Both are bounded here: an unbounded expansion is a crawl, not a survey.

**Not every edge is evolution.** A direct citation with publication order is
provenance that one paper came after another and knew about it. Bibliographic
coupling, co-citation and topical adjacency are *retrieval* relations: useful
for finding candidates, and not evidence that anything evolved. The kinds are
kept apart so a later stage cannot quietly treat proximity as lineage.

Nothing here is scientific evidence. A citation is historical provenance about
papers, and many papers using a method does not support anyone's hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .workid import CanonicalWork, WorkRegistry, parse_identifier

OPENALEX_NS = "openalex"


# Edge kinds. Only the first two carry publication provenance.
DIRECT_CITATION = "DIRECT_CITATION"        # source is cited by target
FORWARD_CITATION = "FORWARD_CITATION"      # target cites source, found by expansion
BIBLIOGRAPHIC_COUPLING = "BIBLIOGRAPHIC_COUPLING"   # two works citing the same sources
CO_CITATION = "CO_CITATION"                        # two works cited together by a third

# SEMANTIC_NEIGHBOR and TEMPORAL_NEIGHBOR were declared here and nothing ever
# built one. A kind no code produces is a claim the runtime cannot keep, and
# the separation below only means something if both sides of it exist. The two
# that remain are computed from reference lists the graph already holds: no
# request, no model call, and no way for either to become lineage.
PROVENANCE_KINDS = frozenset({DIRECT_CITATION, FORWARD_CITATION})
RETRIEVAL_KINDS = frozenset({BIBLIOGRAPHIC_COUPLING, CO_CITATION})
EDGE_KINDS = tuple(sorted(PROVENANCE_KINDS | RETRIEVAL_KINDS))


@dataclass(frozen=True)
class ExpansionBudget:
    """Limits on a survey-sized expansion. Tuned for API latency, not principle."""

    seeds: int = 8
    depth: int = 1
    backward_per_node: int = 12
    forward_per_node: int = 8
    max_nodes: int = 60
    max_degree: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {"seeds": self.seeds, "depth": self.depth,
                "backward_per_node": self.backward_per_node,
                "forward_per_node": self.forward_per_node,
                "max_nodes": self.max_nodes, "max_degree": self.max_degree,
                "basis": "latency_and_budget_not_a_scientific_principle"}


@dataclass(frozen=True)
class CitationEdge:
    """One directed relation between two canonical works."""

    source: str          # the earlier work
    target: str          # the later work that cites it
    kind: str
    reported_by: str = ""      # which index reported the reference
    source_year: str = ""
    target_year: str = ""

    @property
    def is_provenance(self) -> bool:
        return self.kind in PROVENANCE_KINDS

    @property
    def order_known(self) -> bool:
        return bool(self.source_year and self.target_year)

    @property
    def order_ok(self) -> bool:
        """A cited work must not postdate the work citing it."""
        if not self.order_known:
            return True
        try:
            return int(self.source_year) <= int(self.target_year)
        except ValueError:
            return True

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind,
                "reported_by": self.reported_by, "source_year": self.source_year,
                "target_year": self.target_year, "is_provenance": self.is_provenance,
                "publication_order_known": self.order_known,
                "publication_order_ok": self.order_ok}


@dataclass
class EvolutionGraph:
    """Canonical works and the relations between them, with their provenance."""

    registry: WorkRegistry = field(default_factory=WorkRegistry)
    edges: list[CitationEdge] = field(default_factory=list)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def add_work(self, work: Any, *, source: str = "", role: str = "") -> str:
        record = self.registry.add(work, source=source)
        if record is None:
            return ""
        row = work if isinstance(work, Mapping) else getattr(work, "to_dict", dict)()
        stored = self.rows.setdefault(record.canonical_id, {})
        for key, value in dict(row).items():
            if value not in (None, "", [], {}) and not stored.get(key):
                stored[key] = value
        stored["canonical_id"] = record.canonical_id
        stored.setdefault("roles", [])
        if role and role not in stored["roles"]:
            stored["roles"].append(role)
        return record.canonical_id

    def add_edge(self, edge: CitationEdge) -> bool:
        """Record a relation. A cited work that postdates its citer is refused."""
        if not edge.source or not edge.target or edge.source == edge.target:
            return False
        if edge.is_provenance and not edge.order_ok:
            self.blocked.append({"reason": "cited_work_postdates_the_work_citing_it",
                                 "edge": edge.to_dict()})
            return False
        if any(row.source == edge.source and row.target == edge.target
               and row.kind == edge.kind for row in self.edges):
            return False
        self.edges.append(edge)
        return True

    def provenance_edges(self) -> list[CitationEdge]:
        return [row for row in self.edges if row.is_provenance]

    def year_of(self, canonical_id: str) -> str:
        work = self.registry.works.get(canonical_id)
        return work.year if work else ""

    def outgoing(self, canonical_id: str) -> list[CitationEdge]:
        return [row for row in self.edges if row.source == canonical_id]

    def degree(self, canonical_id: str) -> int:
        return sum(1 for row in self.edges
                   if canonical_id in (row.source, row.target))

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for row in self.edges:
            kinds[row.kind] = kinds.get(row.kind, 0) + 1
        return {
            "nodes": len(self.registry.works),
            "edges": len(self.edges),
            "provenance_edges": len(self.provenance_edges()),
            "edge_kinds": kinds,
            "blocked": list(self.blocked),
            **dict(self.telemetry),
            "note": ("A citation is historical provenance about papers. It is not "
                     "scientific evidence and does not support any hypothesis."),
        }


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

# A fetcher takes canonical "namespace:value" keys and returns retrieved rows,
# or a blocked record. Injected so expansion is testable without a socket and so
# a rate-limited source is recorded rather than read as an empty result.
Fetch = Callable[[list[str]], Any]


def _forward_key(row: Mapping[str, Any], canonical: str) -> str:
    """Ask for citers by an OpenAlex id when the source already gave one.

    The forward index takes W-ids. A node keyed by DOI can still be asked
    about, but that costs a resolve request first — which is wasted when the
    W-id arrived with the record.
    """
    for alias in row.get("aliases") or ():
        parsed = parse_identifier(str(alias))
        if parsed and parsed[0] == OPENALEX_NS:
            return f"{OPENALEX_NS}:{parsed[1]}"
    return canonical


def _blocked_record(result: Any) -> dict[str, Any] | None:
    if isinstance(result, Mapping) and result.get("blocked"):
        return dict(result)
    if hasattr(result, "to_dict") and hasattr(result, "reason"):
        return dict(result.to_dict())
    return None


def expand_citations(
    seeds: Iterable[Any],
    *,
    fetch_by_id: Fetch | None = None,
    fetch_citing: Fetch | None = None,
    budget: ExpansionBudget | None = None,
    relevant: Callable[[Mapping[str, Any]], bool] | None = None,
) -> EvolutionGraph:
    """Grow a bounded citation neighbourhood around verified seed papers.

    Backward expansion dereferences references a source already reported, which
    is the step the live path never took: it kept the identifiers as strings and
    compared them. Forward expansion asks which later works cite a seed, when
    the source supports it.

    A source that refuses (rate limit, outage) is recorded as blocked. Reading a
    429 as "this paper has no citations" would silently turn an outage into a
    scientific conclusion about the literature.
    """
    budget = budget or ExpansionBudget()
    graph = EvolutionGraph()
    requests = requested = fetched = verified = 0
    rejected: list[dict[str, Any]] = []

    frontier: list[str] = []
    for work in list(seeds)[: budget.seeds]:
        canonical = graph.add_work(work, role="seed")
        if canonical:
            frontier.append(canonical)

    for _ in range(max(0, int(budget.depth))):
        if len(graph.registry.works) >= budget.max_nodes:
            break
        next_frontier: list[str] = []
        for canonical in frontier:
            if len(graph.registry.works) >= budget.max_nodes:
                break
            row = graph.rows.get(canonical) or {}
            # --- backward: the references this source already reported --------
            wanted = graph.registry.unresolved(row.get("references") or ())
            wanted = wanted[: budget.backward_per_node]
            if wanted and fetch_by_id is not None:
                requests += 1
                requested += len(wanted)
                result = fetch_by_id(list(wanted))
                blocked = _blocked_record(result)
                if blocked:
                    graph.blocked.append({"stage": "backward", "node": canonical, **blocked})
                else:
                    for cited in list(result or []):
                        fetched += 1
                        payload = (cited if isinstance(cited, Mapping)
                                   else getattr(cited, "to_dict", dict)())
                        if relevant is not None and not relevant(payload):
                            rejected.append({"stage": "backward",
                                             "title": str(payload.get("title") or "")[:100],
                                             "reason": "not_relevant_to_the_root_question"})
                            continue
                        if len(graph.registry.works) >= budget.max_nodes:
                            break
                        cited_id = graph.add_work(cited, role="cited")
                        if not cited_id:
                            continue
                        verified += 1
                        if graph.add_edge(CitationEdge(
                                source=cited_id, target=canonical, kind=DIRECT_CITATION,
                                reported_by=str(row.get("source") or ""),
                                source_year=graph.year_of(cited_id),
                                target_year=graph.year_of(canonical))):
                            next_frontier.append(cited_id)
            # --- forward: works that cite this one ---------------------------
            if fetch_citing is not None and graph.degree(canonical) < budget.max_degree:
                requests += 1
                requested += 1
                result = fetch_citing([_forward_key(row, canonical)])
                blocked = _blocked_record(result)
                if blocked:
                    graph.blocked.append({"stage": "forward", "node": canonical, **blocked})
                else:
                    for citing in list(result or [])[: budget.forward_per_node]:
                        fetched += 1
                        payload = (citing if isinstance(citing, Mapping)
                                   else getattr(citing, "to_dict", dict)())
                        if relevant is not None and not relevant(payload):
                            rejected.append({"stage": "forward",
                                             "title": str(payload.get("title") or "")[:100],
                                             "reason": "not_relevant_to_the_root_question"})
                            continue
                        if len(graph.registry.works) >= budget.max_nodes:
                            break
                        citing_id = graph.add_work(citing, role="citing")
                        if not citing_id:
                            continue
                        verified += 1
                        if graph.add_edge(CitationEdge(
                                source=canonical, target=citing_id, kind=FORWARD_CITATION,
                                reported_by=str(payload.get("source") or ""),
                                source_year=graph.year_of(canonical),
                                target_year=graph.year_of(citing_id))):
                            next_frontier.append(citing_id)
        frontier = next_frontier
        if not frontier:
            break

    # Edges among works already in the graph, from references reported but never
    # dereferenced — the case the old pairwise loop was trying to find.
    for canonical, row in list(graph.rows.items()):
        for reference in row.get("references") or ():
            cited_id = graph.registry.resolve(reference)
            if cited_id and cited_id != canonical:
                graph.add_edge(CitationEdge(
                    source=cited_id, target=canonical, kind=DIRECT_CITATION,
                    reported_by=str(row.get("source") or ""),
                    source_year=graph.year_of(cited_id),
                    target_year=graph.year_of(canonical)))

    coupled, cocited = _retrieval_relations(graph)

    graph.telemetry = {
        "budget": budget.to_dict(),
        "bibliographic_coupling_edges": coupled,
        "co_citation_edges": cocited,
        # Three different things, so three names that say which. A live run
        # reported 53 requested and 80 fetched, which reads like a bug and is
        # not one: one forward request returns many rows.
        "expansion_requests": requests,
        "reference_keys_requested": requested,
        "rows_returned_by_sources": fetched,
        "citation_nodes_verified": verified,
        "rejected_nodes": len(rejected),
        "rejected_examples": rejected[:5],
        "blocked_sources": len(graph.blocked),
    }
    return graph


def _retrieval_relations(graph: EvolutionGraph, *, min_shared: int = 2,
                        max_edges: int = 40) -> tuple[int, int]:
    """Relations that help find candidates and never establish lineage.

    Two works citing the same sources are bibliographically coupled; two works
    cited together by a third are co-cited. Both say the literature treats
    these papers as related. Neither says one developed from the other, and
    ``transitions_from_graph`` will not read either of them — which is the
    entire reason the kinds are kept apart.

    Both come free from the reference lists already in the graph: no request,
    no model call, and no full text.
    """
    resolve = graph.registry.resolve
    known = set(graph.rows)
    refs = {node: {resolve(str(ref)) or str(ref)
                   for ref in (row.get("references") or ())}
            for node, row in graph.rows.items()}
    nodes = sorted(refs)

    def link(left: str, right: str, kind: str, reported: str) -> bool:
        first, second = sorted((left, right),
                               key=lambda node: (graph.year_of(node), node))
        return graph.add_edge(CitationEdge(
            source=first, target=second, kind=kind, reported_by=reported,
            source_year=graph.year_of(first), target_year=graph.year_of(second)))

    coupled = 0
    for index, left in enumerate(nodes):
        if coupled >= max_edges:
            break
        for right in nodes[index + 1:]:
            if coupled >= max_edges:
                break
            shared = refs[left] & refs[right]
            if len(shared) >= min_shared and link(
                    left, right, BIBLIOGRAPHIC_COUPLING,
                    f"shared_references:{len(shared)}"):
                coupled += 1

    cocited = 0
    for node in nodes:
        if cocited >= max_edges:
            break
        together = sorted(refs[node] & known)
        for index, left in enumerate(together):
            if cocited >= max_edges:
                break
            for right in together[index + 1:]:
                if cocited >= max_edges:
                    break
                if link(left, right, CO_CITATION, f"cited_together_by:{node}"):
                    cocited += 1
    return coupled, cocited


# ---------------------------------------------------------------------------
# From graph to evolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionBudget:
    """What the transition stage may spend on model calls.

    Expanded nodes are read from title, abstract, year and reported references
    only — no full text is fetched for them. The cost that still bit was
    repetition: the first version re-extracted *both* endpoints of *every*
    edge, so a paper sitting on five edges was extracted five times and a
    24-edge graph could outspend a whole run's call budget by itself.

    Two knobs, because there are exactly two kinds of call here. ``papers``
    caps how many distinct works are extracted; each costs one call and is
    then reused by every edge it touches. ``annotated_edges`` caps how many
    transitions are additionally sent for operator/trigger annotation, which
    is spent on the largest grounded deltas rather than on whatever edge came
    first. Every other transition is still produced — structurally, from the
    cached states, for free.
    """

    papers: int = 10
    annotated_edges: int = 2
    edges: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {"papers": self.papers, "annotated_edges": self.annotated_edges,
                "edges": self.edges,
                "max_model_calls": self.papers + self.annotated_edges,
                "basis": "run_call_budget_not_a_scientific_principle"}


def connected_reading_order(graph: EvolutionGraph, seed_ids: Iterable[str] = ()) -> list[str]:
    """Read the ends of a dated citation before unrelated high-degree nodes.

    Citation order is provenance, not a claim that the cited mechanism caused
    progress. The caller's existing relevance filter and extraction cap apply.
    """
    seeds = set(seed_ids)
    edges = [e for e in graph.provenance_edges()
             if e.order_known and e.order_ok and e.source != e.target
             and e.source in graph.rows and e.target in graph.rows]
    edges.sort(key=lambda e: (-(int(e.source in seeds) + int(e.target in seeds)),
                              e.source, e.target))
    return list(dict.fromkeys(node for edge in edges for node in (edge.source, edge.target)))


class PaperStateCache:
    """One extraction per canonical work, however many edges it appears in.

    The cache is keyed on canonical identity, so the same paper reached as an
    OpenAlex W-id from one index and as a DOI from another is extracted once.
    When the budget is spent, extraction continues without a client rather
    than stopping: that yields a state with no interpreted fields, which
    later refuses to produce a transition. A refusal is the honest outcome of
    an exhausted budget; a guess would not be.
    """

    def __init__(self, registry: Any = None, *, limit: int = 10) -> None:
        import threading

        self.registry = registry
        self.limit = max(0, int(limit))
        self.states: dict[str, Any] = {}
        self.calls = 0
        self.hits = 0
        self.exhausted = 0
        self.failures: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def key(self, value: Any) -> str:
        """Canonical id for a work, an id string, or an extracted state."""
        raw = value
        if not isinstance(raw, str):
            row = (dict(raw) if isinstance(raw, Mapping)
                   else getattr(raw, "to_dict", dict)())
            raw = str(row.get("canonical_id") or row.get("cite_id")
                      or row.get("work_id") or row.get("arxiv_id") or "")
        raw = str(raw)
        if self.registry is not None:
            return self.registry.resolve(raw) or raw
        parsed = parse_identifier(raw)
        return parsed or raw

    def seed(self, key: str, state: Any, *, charged: bool = True) -> None:
        """Adopt a state extracted elsewhere so it is not paid for twice."""
        if not key or state is None or key in self.states:
            return
        self.states[key] = state
        if charged:
            self.calls += 1

    def _extract(self, key: str, row: Any, use: Any) -> Any:
        from .explore import extract_paper_state
        from .llm import LLMUnavailable

        charged = use is not None
        try:
            state = extract_paper_state(row, client=use)
        except (LLMUnavailable, ValueError, KeyError, TypeError) as exc:
            with self._lock:
                self.failures.append({"work": key,
                                      "reason": f"paper_extraction_failed:{exc}"})
            try:
                state = extract_paper_state(row)
            except (ValueError, KeyError, TypeError):
                return None
            charged = False
        with self._lock:
            if key not in self.states:
                self.states[key] = state
                if charged:
                    self.calls += 1
            return self.states[key]

    def get(self, key: str, row: Any, *, client: Any = None) -> Any:
        with self._lock:
            if key in self.states:
                self.hits += 1
                return self.states[key]
            use = client
            if client is not None and self.calls >= self.limit:
                use, self.exhausted = None, self.exhausted + 1
        return self._extract(key, row, use)

    def prefetch(self, items: Iterable[tuple[str, Any]], *, client: Any = None,
                 workers: int = 4) -> None:
        """Extract several works at once.

        One paper's extraction does not depend on another's, and a live call
        can take over a minute, so a survey reading ten papers in sequence
        spends a quarter of an hour on work that has no ordering. The client
        reserves a ledger slot before it hits the wire and detects a duplicate
        recording afterwards, so concurrent calls neither overspend the
        envelope nor charge twice. Budget is allocated before dispatch rather
        than checked inside each worker, so the cap cannot be raced past.
        """
        pending = list({key: row for key, row in items
                        if key and key not in self.states}.items())
        if not pending:
            return
        with self._lock:
            allowed = max(0, self.limit - self.calls) if client is not None else 0
            if client is not None:
                self.exhausted += max(0, len(pending) - allowed)
        plan = [(key, row, client if index < allowed else None)
                for index, (key, row) in enumerate(pending)]
        if allowed < 2 or workers < 2:
            for key, row, use in plan:
                self._extract(key, row, use)
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(workers), len(plan))) as pool:
            list(pool.map(lambda job: self._extract(*job), plan))

    def spend(self) -> dict[str, Any]:
        return {"paper_extractions_with_a_client": self.calls,
                "papers_cached": len(self.states),
                "extractions_reused": self.hits,
                "paper_budget_exhausted": self.exhausted,
                "paper_extraction_failures": self.failures[:5]}


def transitions_from_graph(
    graph: EvolutionGraph,
    *,
    client: Any = None,
    limit: int = 24,
    budget: ExtractionBudget | None = None,
    cache: PaperStateCache | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Extract structured transitions along provenance edges only.

    A retrieval relation — coupling, co-citation, topical adjacency — never
    produces a transition here, however close the two papers look. "Paper B
    came after Paper A and cited it" is the weakest claim that can support
    "B developed from A", and even that is not enough on its own: the extractor
    still requires publication order and a grounded state delta, and refuses
    when either is missing.

    Cost is bounded in two stages. Every endpoint is extracted at most once,
    highest-degree first, so the call that reads a paper is paid for by all
    the edges it sits on. Transitions are then built from those cached states
    with no further calls at all; only the few with the largest grounded
    change are sent back for annotation.
    """
    from .explore import extract_transition

    budget = budget or ExtractionBudget()
    ceiling = min(max(0, int(limit)), max(0, int(budget.edges)))
    if cache is None:
        cache = PaperStateCache(graph.registry, limit=budget.papers)

    edges = [edge for edge in graph.provenance_edges()[:ceiling]
             if graph.rows.get(edge.source) and graph.rows.get(edge.target)]

    # Extract the works that unlock the most edges first: a paper on five
    # edges costs one call and pays for all five.
    demand: dict[str, int] = {}
    for edge in edges:
        for node in (edge.source, edge.target):
            demand[node] = demand.get(node, 0) + 1
    order = connected_reading_order(graph)
    order += [node for node in sorted(demand, key=lambda name: (-demand[name], name)) if node not in order]
    cache.prefetch([(node, graph.rows[node]) for node in order if node in demand],
                   client=client)

    gaps: list[dict[str, Any]] = list(cache.failures)
    structural: list[tuple[Any, CitationEdge]] = []
    for edge in edges:
        left, right = cache.states.get(edge.source), cache.states.get(edge.target)
        if left is None or right is None:
            gaps.append({"source": edge.source, "target": edge.target,
                         "reason": "paper_state_unavailable"})
            continue
        try:
            transition = extract_transition(left, right, client=None,
                                            registry=graph.registry)
        except (ValueError, KeyError, TypeError) as exc:
            gaps.append({"source": edge.source, "target": edge.target,
                         "reason": f"extraction_failed:{exc}"})
            continue
        if getattr(transition, "status", "") != "extracted":
            gaps.append({"source": edge.source, "target": edge.target,
                         "reason": "; ".join(getattr(transition, "reasons", ()) or
                                             ("not_extracted",))})
            continue
        structural.append((transition, edge))

    # Annotation names the operator and the trigger in the papers' own words.
    # It is worth a call on a transition that changed four components and not
    # on one that changed a title.
    ranked = sorted(
        range(len(structural)),
        key=lambda index: (-len(structural[index][0].changed_component),
                           structural[index][1].target_year,
                           structural[index][1].target))
    chosen = (set(ranked[: max(0, int(budget.annotated_edges))])
              if client is not None else set())
    out: list[Any] = []
    annotated = 0
    for index, (transition, edge) in enumerate(structural):
        if index in chosen:
            try:
                richer = extract_transition(cache.states[edge.source],
                                            cache.states[edge.target],
                                            client=client, registry=graph.registry)
            except (ValueError, KeyError, TypeError) as exc:
                gaps.append({"source": edge.source, "target": edge.target,
                             "reason": f"annotation_failed:{exc}"})
                richer = None
            if richer is not None and getattr(richer, "status", "") == "extracted":
                transition, annotated = richer, annotated + 1
        out.append(transition)

    graph.telemetry["extraction"] = {
        # Extractions performed with a client, which is not the same as calls
        # on the wire: a warm replay cache serves some of them for nothing. The
        # ledger is the authority on what was spent, and the resource view
        # reads it.
        "rule": "one-extraction-per-canonical-work-v1",
        "budget": budget.to_dict(),
        "provenance_edges_considered": len(edges),
        # What the previous version would have paid: it extracted both
        # endpoints of every edge, so this is the count of reads a shared
        # cache turned into nothing. Kept separate from `extractions_reused`,
        # which counts genuine cache hits and is legitimately zero when the
        # pre-pass already walked every distinct node.
        "edge_endpoint_lookups": 2 * len(edges),
        "extractions_avoided": max(0, 2 * len(edges) - len(cache.states)),
        "transitions_extracted": len(out),
        # Retrieval drops a transition whose operator could not be named, so
        # the count of extracted transitions overstates what the lane can act
        # on. Both numbers are reported rather than the flattering one.
        "transitions_with_named_operator": sum(
            1 for row in out if getattr(row, "operator", "unknown") != "unknown"),
        "annotation_calls": annotated,
        "extractions_and_annotations": cache.calls + annotated,
        "full_text_fetched_for_expanded_nodes": False,
        **cache.spend(),
    }
    return out, gaps


def recover_trajectories(
    graph: EvolutionGraph,
    transitions: Iterable[Any],
    *,
    max_length: int = 5,
) -> list[dict[str, Any]]:
    """Chain transitions into the paths they actually form.

    An isolated A→B is one step. What a research lineage looks like is
    A→B→C: an assumption removed, then a representation changed, then a new
    evaluator. The operator sequence along a path is the motif worth reusing,
    and a single edge cannot show one.
    """
    by_source: dict[str, list[Any]] = {}
    ids: dict[int, tuple[str, str]] = {}
    for transition in transitions:
        left = graph.registry.resolve(getattr(transition, "source_cite_id", ""))
        right = graph.registry.resolve(getattr(transition, "target_cite_id", ""))
        if not left or not right:
            continue
        ids[id(transition)] = (left, right)
        by_source.setdefault(left, []).append(transition)

    paths: list[dict[str, Any]] = []

    def walk(node: str, chain: list[Any], seen: set[str]) -> None:
        extended = False
        if len(chain) < max_length:
            for transition in by_source.get(node, ()):
                _, right = ids[id(transition)]
                if right in seen:
                    continue
                extended = True
                walk(right, chain + [transition], seen | {right})
        if not extended and len(chain) >= 1:
            states = [ids[id(chain[0])][0]] + [ids[id(row)][1] for row in chain]
            paths.append({
                "states": states,
                "length": len(chain),
                "operator_sequence": [getattr(row, "operator", "") for row in chain],
                "transitions": [getattr(row, "to_dict", dict)() for row in chain],
                "years": [graph.year_of(node) for node in states],
                "provenance": "verified_citation_path",
            })

    for start in sorted(by_source):
        walk(start, [], {start})
    # Longest first: a motif of several steps says more than a single hop.
    paths.sort(key=lambda row: (-row["length"], row["states"][0]))
    seen_chains: set[tuple[str, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in paths:
        key = tuple(row["states"])
        if any(set(key) <= set(other) and key != other for other in seen_chains):
            continue
        seen_chains.add(key)
        unique.append(row)
    return unique
