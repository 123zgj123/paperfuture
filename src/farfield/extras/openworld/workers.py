"""Single-action adapters over the mature research workers.

The controller owns scheduling and redesign. These functions never enter a
mission, synthesize arm numbers, or retry a scientific result.
"""
from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping

from .. import chain
from ..brief import compile_brief
from ..diagnose import diagnosis_from_payload, judge_probe, write_diagnosis
from ..evidence import ablation_required, evidence_id, sha256_text
from ..generate import GeneratedCard, GenerationRefused, generate_card
from ..hostexp import ExecuteError, execute_protocol
from ..knowledge import harvest_survey
from ..livefeed import CompositeFeed, FreshWork
from ..llm import LLMUnavailable
from ..packet import render_protocol
from ..prior import strongest_prior
from ..probeexp import floor_tier, probe_timeout, resolve_tier, run_probe, write_probe
from ..reframe import generate_reframe
from ..validity import assess_probe, sample_floor
from ..world import bind_world, digest_files, load_fixture, reads_world_data
from ..workspace import write_idea
from .actions import PROBE, SURVEY, SYNTHESIZE, THEORIZE, VERIFY
from .events import make_event
from .kernel import CandidateResult, readonly_view
from .worldio import is_attested


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bounded_survey_extraction(budget: Any, client: Any, *, opening: bool = False,
                               source_check: bool = False, map_revision: bool = False) -> Any:
    """Keep a small survey from spending the calls needed to develop its ideas."""
    if opening:
        # Map enough prior work to choose a question. Deep reading follows a
        # named hypothesis or objection, rather than exhausting the first turn.
        budget = replace(budget, papers=min(budget.papers, 2), annotated_edges=0)
    ledger = getattr(client, "ledger", None)
    remaining = getattr(ledger, "remaining", None)
    if not callable(remaining):
        return budget
    calls = remaining().get("api_calls")
    if (isinstance(calls, bool) or not isinstance(calls, (int, float))
            or not math.isfinite(calls)):
        return budget
    # Search has already spent its call. Reserve discovery and critique, plus
    # the relevance screening performed after this extraction stage.
    reserve = 2 if opening or source_check or map_revision else max(4, math.ceil(calls / 2))
    available = max(0, math.floor(calls - reserve - 1 - int(source_check) - int(map_revision)))
    papers = min(max(0, int(budget.papers)), available)
    if opening and papers < 2:
        papers = 0  # an isolated read cannot establish the requested transition
    annotations = min(max(0, int(budget.annotated_edges)), available - papers)
    return replace(budget, papers=papers, annotated_edges=annotations)


def _reported_sample(metrics: Mapping[str, Any], world: Any) -> dict[str, Any]:
    """How many units the registered experiment says each arm measured.

    Nothing else in the runtime knows this. The world's size is an upper bound,
    not the sample: an experiment on a 150-row table may have measured twenty
    rows, and reading the bound as the sample would overstate power. So only the
    experiment's own report counts — and a report larger than the attested world
    holds is incoherent and refused rather than believed.
    """
    row = dict(metrics or {})
    values = [row.get("n_treatment"), row.get("n_control")]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        return {"sample_size": None,
                "sample_size_basis": "not_reported_by_the_registered_experiment"}
    smallest = min(int(v) for v in values)
    scale = 0
    if world is not None:
        try:
            from .worldio import summarize

            scale = max((int(v) for key, v in summarize(world).items()
                         if key.startswith("n_") and isinstance(v, int)), default=0)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            scale = 0
    if scale and max(int(v) for v in values) > scale:
        return {"sample_size": None,
                "sample_size_basis": "refused_report_exceeds_the_attested_world",
                "reported_sample_sizes": [int(v) for v in values],
                "world_scale": scale}
    return {"sample_size": smallest,
            "sample_size_basis": "reported_by_the_registered_experiment",
            "reported_sample_sizes": [int(v) for v in values],
            "world_scale": scale or None}


def _project_brief(kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    """The controller's compact project brief, as worker context.

    ``project_for`` used to compute this on every action and hand it to workers
    that never read it, so a long run's prompts were built from accumulated
    narrative while the relevant slice was discarded. The brief is a derived
    projection: it constrains what the worker is asked to do, and it is not a
    source of scientific facts.
    """
    projection = kwargs.get("projection")
    if not isinstance(projection, Mapping):
        return ()
    journal = projection.get("research_journal")
    if not isinstance(journal, Mapping):
        return ()
    brief = {
        "research_contract": journal.get("research_contract"),
        "current_strategy": journal.get("strategy"),
        "next_decisive_action": journal.get("next_decisive_action"),
        "open_objections_you_must_not_answer_with_wording": journal.get("open_objections"),
        "relevant_world_evidence": journal.get("relevant_world_evidence"),
        "experiment_design_feedback": journal.get("experiment_design_feedback"),
        "exploration_environment": journal.get("exploration_environment"),
        "remaining_resources": journal.get("resources"),
        "epistemic_reminders": journal.get("epistemic_reminders"),
    }
    if projection.get("clean_restart"):
        restart = dict(projection["clean_restart"])
        brief["clean_restart"] = {
            "reason": restart.get("reason"),
            "root_question": restart.get("root_question"),
            "failure_lessons": restart.get("failure_lessons"),
            "dropped": restart.get("dropped"),
            "note": restart.get("note"),
        }
    if projection.get("contingency_plan"):
        brief["outcome_contingencies"] = projection["contingency_plan"]
    return (json.dumps({"project_brief_derived_from_scientific_state": brief},
                       ensure_ascii=False),)


def _names_observable(declared: str, registered: str) -> bool:
    """Does this declared observable name the registered one?

    Not string equality between two independently written texts. The proposal
    may describe the measurement — "dropout-induced petal-minus-sepal
    mean_pairwise_distance contrast" — and what matters is that it names the
    registered identifier the probe actually measured, as a whole token.
    """
    import re

    left, right = str(declared or "").strip(), str(registered or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(right)}(?![A-Za-z0-9_])", left) is not None


def _card(raw: Any) -> GeneratedCard:
    if isinstance(raw, GeneratedCard):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("no generated card is attached to this theory")
    allowed = {item.name for item in fields(GeneratedCard)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for name in ("pair", "pair_nodes"):
        if name in values:
            values[name] = tuple(values[name])
    result = GeneratedCard(**values)
    if not all((result.card_id, result.claim, result.mechanism, result.prediction)):
        raise ValueError("the generated card is incomplete")
    if Path(result.card_id).name != result.card_id or result.card_id in {".", ".."}:
        raise ValueError("invalid card id")
    return result


def _discovery_idea(state: Mapping[str, Any], idea_id: str = "") -> dict[str, Any]:
    """A discovery idea as compile context for a registered hypothesis.

    Named, it is the idea this hypothesis compiles. Unnamed, it is the selected
    idea, and only when discovery judged it promising. Either way it is a
    GENERATED proposal handed to the model, never evidence.
    """
    from .discovery import PROMISING, discovery_view

    view = discovery_view(state)
    rows = {str(row["id"]): row for row in view["ideas"]}
    row = rows.get(idea_id) if idea_id else rows.get(view["selected_idea"])
    if row is None:
        return {}
    feedback = view["feedback"].get(str(row["id"])) or {}
    if not idea_id and feedback.get("verdict") != PROMISING:
        return {}
    return {"idea_id": row["id"], "idea": dict(row.get("idea") or {}),
            "verdict": feedback.get("verdict"),
            "strongest_competing_explanation": feedback.get("strongest_competing_explanation"),
            "required_evidence": list(feedback.get("required_evidence") or []),
            "epistemic": "GENERATED",
            "note": ("a discovery idea to compile into a world-bound registered hypothesis; "
                     "a proposal, not evidence")}


class LegacyResearchWorkers:
    """Inject a replay/live client and feed; no network work at construction.

    Executors accept ``(action, state, **kwargs)``. ``world`` may be an
    explicit fixture; otherwise it is loaded from workspace/worlds/<id>.
    ProbeEvidence carries ``card``, ``diagnosis``, ``protocol_dir``, and
    immutable registration identities for subsequent VERIFY/SYNTHESIZE.
    """

    def __init__(self, client=None, feed=None, workspace=None, *, knowledge_client=None):
        self.client = client
        self.feed = feed
        self.workspace = Path(workspace) if workspace is not None else None
        if knowledge_client is not None and (getattr(client, 'ledger', None) is None
                or getattr(knowledge_client, 'ledger', None) is not client.ledger):
            raise ValueError('knowledge_model_requires_shared_budget_ledger')
        self.knowledge_client = knowledge_client

    def executors(self):
        return {SURVEY: self.survey, THEORIZE: self.theorize, PROBE: self.probe,
                VERIFY: self.verify, SYNTHESIZE: self.synthesize}

    def _workspace(self, kwargs):
        value = kwargs.get("workspace") or self.workspace
        if value is None:
            raise ValueError("a workspace is required to preserve research artifacts")
        return Path(value)

    def _failure(self, action, reason, *, status="blocked", failed=False, **extra):
        if isinstance(reason, LLMUnavailable):
            return self._service_unavailable(action, reason)
        payload = {"reason": str(reason), "theory_id": action.target,
                   "question_id": action.frontier_target_id, **extra}
        if action.action_type == SYNTHESIZE:
            # A report could not be produced. This neither makes its question
            # unanswerable nor invalidates the evidence being reported.
            return CandidateResult(status=status, events=[], extras=payload)
        # A worker refusal, malformed proposal, missing file or disconnected
        # model describes this attempt. Only the capability/acquisition layer
        # can establish a missing resource; these errors cannot close a question.
        return CandidateResult(status=status, events=[make_event(
            "ActionFailed", payload,
            action_type=action.action_type, frontier_target_id=action.frontier_target_id or action.target)],
            extras=payload)

    def _service_unavailable(self, action, error):
        """The model service did not answer. That is about the run, not the question.

        Every handler here used to report it as the action being blocked, on the
        action's question — so a spent API balance was recorded as a research
        question that could not be answered, and the run ended BLOCKED. No event
        is proposed: nothing is learned about the question, and no design failed.
        The controller records the attempt, and the resource view decides whether
        the service has run out.
        """
        from ..llm import quota_exhausted
        from .resources import local_model_budget_meter

        text = str(error)
        meter = local_model_budget_meter(error)
        if meter:
            return CandidateResult(status='budget_exhausted', events=[], extras={
                'reason': text, 'budget_meter': meter, 'budget_basis': 'typed_client_envelope',
                'theory_id': action.target, 'question_id': action.frontier_target_id})
        payload = {"reason": text,
                   "service": str(getattr(error.record, "missing_capability", "") or "llm_endpoint"),
                   "quota_exhausted": quota_exhausted(text),
                   "theory_id": action.target, "question_id": action.frontier_target_id}
        return CandidateResult(status="service_unavailable", events=[], extras=payload)

    def _theory(self, action, state):
        row = next((item for item in state.get("theories", [])
                    if item.get("id") == action.target), None)
        if row is None and action.extra.get("theory"):
            row = action.extra["theory"]
        return dict(row or {})

    def _works(self, state, question_id=""):
        result = {}
        for artifact in state.get("artifacts", []):
            if artifact.get("kind") != "literature_survey":
                continue
            if question_id and artifact.get("question_id") not in {"", question_id}:
                continue
            for row in artifact.get("works", []):
                work = FreshWork.from_dict(row)
                if work.cite_id():
                    result[work.cite_id()] = work
        return list(result.values())

    def _world(self, state, kwargs):
        supplied = kwargs.get("world")
        world_id = str(state.get("world_id") or kwargs.get("world_id") or "")
        if supplied is not None:
            world = load_fixture(Path(supplied.root))
            if world.digest != supplied.digest or world.id != supplied.id:
                raise ValueError("world identity differs from its manifest")
        else:
            if not world_id or Path(world_id).name != world_id or world_id in {".", ".."}:
                raise ValueError("no attested world is bound")
            world = load_fixture(self._workspace(kwargs) / "worlds" / world_id)
        if world_id and world.id != world_id:
            raise ValueError("world is not the active controller world")
        if not is_attested(world):
            raise ValueError("world is generated or its freeze cannot be attested")
        return world

    def survey(self, action, state, **kwargs):
        state = readonly_view(state)
        topic = str(state.get("goal") or action.target)
        question = next((row for row in state.get("open_questions", [])
                         if row.get("id") == action.target), {})
        try:
            feed = self.feed if self.feed is not None else CompositeFeed()
            # Ask the index for the subject, not for the question's wording.
            # A phrase builder working in sentence order turned this question
            # into `all:"does petal"`, and the survey came back with conic
            # programming. What the phrase changes is only what is asked for:
            # every result is still verified and still screened for subject.
            from .exploration import search_topic

            root_question = str(state.get("contract", {}).get("root_question") or topic)
            inquiry = action.extra.get("knowledge_request") or {}
            inquiry_question = str(inquiry.get("question") or "")
            from .understanding import technical_map, can_spend
            maps = [r['technical_map'] for r in state.get('artifacts', [])
                    if r.get('kind') == 'literature_survey' and r.get('technical_map')
                    and int(r.get('project_epoch') or 0) == int(state.get('project_epoch') or 0)]
            sketch, map_gap = {}, ''
            if self.knowledge_client is not None and not maps and can_spend(self.client, 8 + int(bool(inquiry))):
                try:
                    sketch = technical_map(self.knowledge_client, inquiry_question or root_question)
                except (ValueError, TypeError, KeyError, LLMUnavailable) as exc:
                    map_gap = type(exc).__name__
            elif self.knowledge_client is not None and not maps:
                map_gap = 'budget_reserved_for_reading_and_followup'
            search_questions = sketch.get('search_questions') or []
            search = search_topic(inquiry_question or (search_questions[0] if search_questions else root_question), str(question.get("text") or ""),
                                  client=self.client)
            query_topic = str(search.get("topic") or topic)
            harvest = harvest_survey(feed,
                                     query_topic, str(question.get("text") or query_topic),
                                     topic=query_topic,
                                     claim=str(action.extra.get("claim") or ""),
                                     fetcher=kwargs.get("fetcher"))
            if not harvest.works:
                return self._failure(action, "no_verified_literature", citations=[], harvest=harvest.to_dict())
            works = list(harvest.works)
            documents, document_gaps = {}, []
            reader = getattr(feed, "read_work", None)
            if callable(reader):
                from ..livefeed import FeedBlocked
                for work in works[:6]:
                    document = reader(work)
                    if isinstance(document, FeedBlocked):
                        document_gaps.append(document.to_dict())
                    elif (isinstance(document, dict) and work.arxiv_id
                          and document.get("arxiv_id") == work.arxiv_id):
                        documents[work.cite_id()] = document
            prior = strongest_prior(str(action.extra.get("claim") or question.get("text") or topic), works)
            spans = [{"cite_id": work.cite_id(), "field": "abstract", "start": 0,
                      "end": len(work.abstract), "text": work.abstract, "url": work.href()}
                     for work in works if work.abstract]
            identity = sha256_text(json.dumps(harvest.to_dict(), sort_keys=True))[:20]
            path = self._workspace(kwargs) / "literature" / f"{identity}.json"
            artifact = {"id": f"survey-{identity}", "kind": "literature_survey",
                        "knowledge_request": dict(inquiry),
                        "project_epoch": int(state.get('project_epoch') or 0),
                        "technical_map_draft": sketch, "technical_map_gap": map_gap,
                        "theory_id": action.extra.get("theory_id", ""),
                        "question_id": action.target, "citations": [work.cite_id() for work in works],
                        "read_first": [work.cite_id() for work in works], "spans": spans,
                        "prior": prior.to_dict() if prior else None, "path": str(path),
                        "source_documents": documents, "document_gaps": document_gaps,
                        "search_topic": search,
                        **harvest.to_dict()}
            if action.extra.get("extract_trajectories"):
                # Keyword search finds seeds. A citation graph is built by
                # dereferencing the references a source already reported, which
                # the previous pairwise loop never did: it looked for citations
                # *inside* one search's six results, and in six live segments
                # found none.
                from ..citations import (
                    ExpansionBudget, ExtractionBudget, PaperStateCache,
                    connected_reading_order, expand_citations, recover_trajectories, transitions_from_graph,
                )

                gaps: list[dict[str, Any]] = []
                root_question = str(state.get("contract", {}).get("root_question") or topic)

                def relevance_of(row: Mapping[str, Any]) -> bool:
                    """Expansion follows citations that bear on the question.

                    An unbounded neighbourhood is a crawl. Off-topic nodes are
                    counted and reported rather than followed, so a survey can
                    say whether retrieval or expansion was the limit.
                    """
                    from .exploration import paper_relevance

                    return paper_relevance(inquiry_question or root_question, row)["relevance"] != "off_topic"

                budget = ExpansionBudget(**dict(action.extra.get("expansion_budget") or {}))
                graph = expand_citations(
                    [w.to_dict() for w in works[: budget.seeds]],
                    fetch_by_id=getattr(feed, "works_by_id", None),
                    fetch_citing=getattr(feed, "works_citing", None),
                    budget=budget,
                    relevant=relevance_of,
                )
                # The anchor papers *are* graph seeds. Running both stages
                # through one cache means an edge touching a seed costs no
                # further call, and one budget covers the whole survey rather
                # than two stages each spending freely: the previous version
                # extracted every edge's two endpoints again, so a paper on
                # five edges was read six times over.
                extraction = ExtractionBudget(
                    **dict(action.extra.get("extraction_budget") or {}))
                extraction = _bounded_survey_extraction(extraction, self.client,
                    source_check=bool(inquiry),
                    map_revision=self.knowledge_client is not None
                        and can_spend(self.client, 6 + int(bool(inquiry))),
                    opening=not state.get('theories') and not any(
                        r.get('kind') == 'discovery_idea' for r in state.get('artifacts', [])))
                cache = PaperStateCache(graph.registry, limit=extraction.papers)
                seeds = [(cache.key(work.cite_id()),
                          documents.get(work.cite_id(), work)) for work in works[:6]]
                pair = connected_reading_order(graph, [key for key, _ in seeds])[:2]
                cache.prefetch([(key, documents.get(key, graph.rows[key])) for key in pair], client=self.client)
                cache.prefetch(seeds, client=self.client)
                papers = [state_of for state_of in
                          (cache.states.get(key) for key, _ in seeds)
                          if state_of is not None]
                extracted, transition_gaps = transitions_from_graph(
                    graph, client=self.client, budget=extraction, cache=cache)
                gaps.extend(transition_gaps)
                transitions = [t.to_dict() for t in extracted]
                trajectories = recover_trajectories(graph, extracted)
                # Anchor the research state on papers that bear on the question.
                # A live run anchored an iris-morphometrics question to a paper on
                # first-order conic programming because both mention "geometry",
                # and the generated hypothesis then defended itself against a
                # flower-reconstruction paper. Off-topic retrievals are recorded,
                # not used to frame the science.
                from .exploration import relevance_report

                root = str(state.get("contract", {}).get("root_question") or topic)
                relevance = relevance_report(inquiry_question or root, [
                    {"cite_id": p.cite_id, "title": p.title, "abstract": p.abstract,
                     "problem": p.problem, "research_question": p.research_question}
                    for p in papers], client=self.client)
                usable = set(relevance["usable"])
                anchoring = [p for p in papers if p.cite_id in usable] or []
                anchor = {"problem": topic, "research_question": str(question.get("text") or topic),
                          "epistemic": "GENERATED",
                          "anchor_basis": [p.cite_id for p in anchoring],
                          "retrieval_relevance": relevance,
                          "off_topic_not_anchored": relevance["off_topic"],
                          "unresolved_questions": [r.get("text") for r in state.get("open_questions", [])],
                          "actual_evidence": list(state.get("evidence_records", []))}
                for name in ("bottleneck_gap", "assumptions", "representation", "evidence"):
                    anchor[name] = list(dict.fromkeys(
                        getattr(p, name) for p in anchoring if getattr(p, name) != "unknown"))
                if relevance["gap"]:
                    gaps.append({"reason": relevance["gap"],
                                 "retrieved": len(papers),
                                 "note": "the anchor is empty rather than built from "
                                         "papers that only share a word"})
                artifact.update(paper_states=[p.to_dict() for p in papers],
                                connected_reading={"pair_ids": pair,
                                    "status": "pair_requested" if len(pair) == 2 and extraction.papers >= 2
                                              else "under_assessed_not_refuted",
                                    "budget": extraction.to_dict(),
                                    "basis": "dated_citation_not_causal_attribution"},
                                trajectory_sources=[{"cite_id": key,
                                    "title": graph.rows[key].get("title", ""),
                                    "abstract": graph.rows[key].get("abstract", ""),
                                    "published": graph.rows[key].get("published", ""),
                                    "identity_scope": "citation_index_metadata"} for key in pair],
                                research_transitions=transitions,
                                research_trajectories=trajectories,
                                citation_graph=graph.summary(),
                                research_state=anchor, extraction_gaps=gaps,
                                retrieval_relevance=relevance)
            from .sourcecheck import check_sources, reading_sources
            # Both readers receive the same saved source excerpts. Previously
            # full text was fetched and archived but discarded at this boundary.
            related = [{**row, 'work_id': row['cite_id']}
                       for row in artifact.get('trajectory_sources') or []]
            material = reading_sources([w.to_dict() for w in works[:6]] + related, documents)
            artifact['reading_sources'] = material
            if self.knowledge_client is not None and can_spend(self.client, 3 + int(bool(inquiry))):
                try:
                    artifact['technical_map'] = technical_map(self.knowledge_client, root_question,
                        sources=material, previous=sketch or (maps[-1] if maps else None))
                except (ValueError, TypeError, KeyError, LLMUnavailable) as exc:
                    artifact['technical_map_gap'] = type(exc).__name__
            if inquiry and self.client is not None:
                try:
                    artifact['source_check'] = check_sources(self.client,
                        str(action.extra.get('claim') or inquiry_question), material)
                except (ValueError, TypeError, KeyError, LLMUnavailable) as exc:
                    # Preserve sources even when interpretation fails. Never
                    # serialize a transport error that might contain a key.
                    artifact['source_check_gap'] = type(exc).__name__
            _json(path, artifact)
            return CandidateResult(status="surveyed", events=[make_event(
                "ArtifactAdded", artifact, action_type=SURVEY,
                frontier_target_id=action.frontier_target_id or action.target)], extras=artifact)
        except (OSError, ValueError, LLMUnavailable) as exc:
            return self._failure(action, exc, citations=[])

    def theorize(self, action, state, **kwargs):
        state = readonly_view(state)
        if action.extra.get("research_lane"):
            return self._research_branch(action, state, **kwargs)
        if self.client is None:
            return self._failure(action, "research_model_not_connected")
        topic = str(state.get("goal") or action.target)
        works = self._works(state, action.target)
        question = next((row for row in state.get("open_questions", []) if row.get("id") == action.target), {})
        knowledge = tuple(str(row.get("text") or row.get("reason") or "")
                          for row in state.get("failed_designs", []))
        knowledge += _project_brief(kwargs)
        context = tuple(f"{work.cite_id()}: {work.title}. {work.abstract}" for work in works)
        common = dict(seed_label=topic, near_labels=(topic,), topic=topic, context=context,
                      knowledge=knowledge, program=(str(question.get("text") or action.target),))
        try:
            if action.extra.get("far_labels"):
                card = generate_card(self.client, **common,
                    far_labels=tuple(action.extra["far_labels"]),
                    operator=str(action.extra.get("operator") or "directional"),
                    alienness=float(action.extra.get("alienness") or 0.0),
                    label_to_node=dict(action.extra.get("label_to_node") or {}),
                    world_menu=action.extra.get("world_menu"))
            else:
                card = generate_reframe(self.client, **common,
                                        operator=str(action.extra.get("operator") or "assumption_removal"))
            competing = list(dict.fromkeys([card.objection] + list(state.get("competing_explanations", []))))
            competing = [text for text in competing if text]
            theory = {"id": card.card_id, "question_id": action.target, "target": action.target,
                      "text": card.claim, "mechanism": card.mechanism,
                      "assumptions": [card.pair[1]], "predictions": [card.prediction],
                      "boundary_conditions": [], "competing": competing,
                      "discriminating_predictions": [{"prediction": card.prediction, "maps_to": PROBE,
                          "competitor": competing[0] if competing else "", "competitor_differs": False}],
                      "card": card.to_dict(), "epistemic": "GENERATED",
                      "evidence_considered": [row.get("evidence_id") for row in state.get("evidence_records", [])],
                      "read_first": [work.cite_id() for work in works]}
            return CandidateResult(status="theorized", events=[make_event(
                "TheoryCreated", theory, action_type=THEORIZE,
                frontier_target_id=action.frontier_target_id or action.target)], extras={"theory": theory})
        except (GenerationRefused, LLMUnavailable, ValueError, KeyError) as exc:
            return self._failure(action, exc)

    def _research_branch(self, action, state, **kwargs):
        """Apply a historical operator or explore a different lane, once per action."""
        from ..explore import copy_check, retrieve_transitions
        if self.client is None:
            return self._failure(action, "research_model_not_connected")
        works = self._works(state, action.target)
        if not works:
            return self._failure(action, "verified_literature_required_before_theory")
        lane = action.extra["research_lane"]
        parent_id = action.extra.get("parent_theory_id", "")
        parent = next((t for t in state.get("theories", []) if t.get("id") == parent_id), {})
        hop = int(parent.get("hop") or 1) + 1 if parent_id else 1
        requested_hop = action.extra.get("hop", hop)
        if (type(requested_hop) is not int or requested_hop != hop
                or parent_id and (not parent or parent.get("question_id") != action.target)):
            return self._failure(action, "parent_lineage_or_hop_mismatch")
        # Re-read canonical state, never trust caller-supplied evidence_context.
        # One definition of "the parent's admitted evidence", shared with
        # operator selection so the hop gate and the choice of transformation
        # cannot disagree about what exists.
        from .exploration import admitted_evidence_for

        evidence = admitted_evidence_for(state, parent_id)
        failures = [r for r in state.get("failed_designs", []) if r.get("theory_id") == parent_id]
        revision_id = str(action.extra.get("revision_id") or "")
        if hop > 1 and not (evidence or failures or revision_id):
            return self._failure(action, "next_hop_requires_actual_parent_evidence_or_explicit_failure")
        surveys = [r for r in state.get("artifacts", []) if r.get("kind") == "literature_survey"
                   and r.get("question_id") in {"", action.target}]
        transitions = [t for r in surveys for t in r.get("research_transitions", [])]
        anchor = dict(next((r["research_state"] for r in reversed(surveys) if r.get("research_state")), {}))
        anchor.update(problem=state.get("goal"), actual_evidence=evidence,
                      contradiction=state.get("contradictions", []), failed_designs=failures)
        if parent.get("assumptions"):
            anchor["assumptions"] = parent["assumptions"]
        selected = retrieve_transitions(anchor, transitions)
        policy = next((r.get("policy") for r in reversed(state.get("artifacts", []))
                       if r.get("kind") == "research_policy_snapshot"), None)
        if policy and selected:
            from ..routing import preferred_operators
            order = preferred_operators(policy, tuple(dict.fromkeys(t.operator for t in selected)))
            selected.sort(key=lambda t: order.index(t.operator))
        if evidence and any(r.get("outcome") in {"weakens", "negative"} for r in evidence):
            selected = [t for t in selected if t.operator != parent.get("operator_reused")]
        lane_fallback = ""
        if lane == "trajectory_supported" and not selected:
            if not parent_id:
                return self._failure(action, "no_applicable_grounded_trajectory", research_lane=lane)
            # An evidence-driven pivot may leave a depleted historical library.
            # Record the substitution: a lane that quietly becomes another lane
            # is how a system keeps reporting a capability it stopped using.
            lane_fallback = "trajectory_supported_had_no_applicable_transition"
            lane = "assumption_reframe"
        chosen = selected[:2] if lane == "trajectory_supported" else []
        # A promising discovery idea is what this hypothesis compiles, when it
        # names one; otherwise the selected promising idea is context only.
        bridge = str(action.extra.get("discovery_idea_id") or "")
        idea_context = _discovery_idea(state, bridge)
        if bridge and not idea_context:
            return self._failure(action, "discovery_idea_missing", research_lane=lane)
        if chosen:
            # A grounded historical transition names its own operator.
            operator = chosen[0].operator
            selection = {"operator": operator, "basis": "grounded_historical_transition",
                         "considered": [t.operator for t in selected]}
        elif bridge:
            # The compile uses the same generator as the assumption lane; what
            # differs is what it compiles, which the theory records.
            from .exploration import select_operator
            from .state import ScientificState

            generation = select_operator(
                ScientificState.from_dict(dict(state)), lane="assumption_reframe",
                question_id=action.target, parent=parent, policy=policy)
            operator = str(generation["operator"])
            selection = {"operator": operator, "basis": "compiling_a_promising_discovery_idea",
                         "discovery_idea_id": bridge, "generation": generation}
        elif revision_id:
            # A repair hop already knows what has to change; the revision named
            # it. Selecting freely here would discard the diagnosis.
            operator = str(action.extra.get("repair_operator") or "representation_change")
            selection = {"operator": operator, "basis": "carrying_out_design_revision",
                         "revision_id": revision_id,
                         "repair": action.extra.get("repair"),
                         "changed_components": list(action.extra.get("changed_components") or [])}
        else:
            # Without one, the transformation is *selected* from what this state
            # admits, what this project's own credit prefers, and what has not
            # already been asked of this question. It used to be a constant, so
            # two runs of the same question could not differ and the installed
            # policy — which only reordered retrieved operators — touched nothing.
            from .exploration import select_operator
            from .state import ScientificState

            selection = select_operator(
                ScientificState.from_dict(dict(state)), lane=lane,
                question_id=action.target, parent=parent, policy=policy)
            operator = selection["operator"]
        schema = {
            "assumptions": ["explicit unresolved assumption"],
            "competing_explanations": [{"id": "alternative", "claim": "substantive alternative mechanism",
                                        "expected_outcome": {"observable": "metric", "direction": "unchanged"}}],
            "closest_prior_work": ["COPY verified citation id"],
            "inherited_components": ["component inherited from closest prior"],
            "genuinely_changed_components": ["new mechanism or boundary, not only new topic"],
            "claim_delta": "falsifiable claim beyond prior", "mechanism_delta": "specific mechanism change",
            "why_not_reconstruction": "why this is not historical target reconstruction",
            "experiment_proposal": {"hypotheses_compared": ["self", "alternative"],
                "expected_outcomes": {"self": {"observable": "metric", "direction": "decrease"},
                                      "alternative": {"observable": "metric", "direction": "unchanged"}},
                "nuisance_factors": ["measurement confound"], "minimal_discriminating_intervention": "matched intervention",
                "interpretation_if_positive": "which explanation weakens", "interpretation_if_negative": "which claim weakens",
                "interpretation_if_inconclusive": "what remains unknown", "expected_scientific_gain": 0.5,
                "transfer_value": 0.5, "frontier_unlock": 0.5, "cost": 0.1, "risk": 0.1},
        }
        repair_note = ""
        if revision_id:
            repair_note = (
                "\nThis hypothesis carries a recorded design revision. Root cause: "
                f"{action.extra.get('failure_root_cause')!r}. What must change: "
                f"{list(action.extra.get('changed_components') or [])}. Abandon early if: "
                f"{action.extra.get('early_kill_condition')!r}. The revised design must "
                "differ in that named component; restating the previous design with new "
                "wording does not repair anything.")
        instructions = ("In addition to the worker schema include a concrete falsifier and research_metadata following "
            "this shape (examples are schema only, NOT supplied scientific facts): " + json.dumps(schema) +
            "\nNo arbitrary confidence, copying a target paper or apply-X-to-Y-only claim. "
            "Describe competing predictions over the SAME observable and a minimal distinguishing intervention. "
            "REFERENCE INTEGRITY: the focal hypothesis ID is exactly 'self'. Each competing_explanations.id "
            "must be unique and must not be 'self'. hypotheses_compared must contain 'self' and the exact "
            "competitor IDs; expected_outcomes keys must exactly equal hypotheses_compared. Never replace "
            "an ID with a descriptive name or leave the example 'alternative' after renaming its competitor. "
            "Directions must be increase/decrease/unchanged. Priority factors are proposal estimates, not evidence. "
            "For research depth, include research_obligations in research_metadata: separately document "
            "novelty (closest-prior overlap and the exact unestablished delta), importance (who benefits and "
            "which real observations would establish prevalence), feasibility (available inputs versus missing "
            "execution capabilities), and value (strong baselines, equal-cost comparison, transfer conditions "
            "and a kill criterion). These are GENERATED obligations, not ratings or conclusions. "
            "Also include study_design: primary estimand, randomized arms or justified observational scope, "
            "negative control, leakage-safe split, uncertainty analysis, and the smallest informative pilot. "
            "Do not claim novelty from distance or claim effectiveness from your own assessment. "
            "A narrow executable diagnostic must not replace the original research question."
            + repair_note)
        transform = [{"source_state": t.source_state.to_dict(), "operator": t.operator,
                      "changed_component": t.changed_component, "applicability_conditions": t.applicability_conditions,
                      "known_failure_conditions": t.known_failure_conditions} for t in chosen]
        context_blob = {"anchor": anchor, "parent_hypothesis": parent,
            "actual_parent_evidence": evidence, "execution_failures_not_evidence": failures,
            "transferable_operators": transform,
            "literature_read_spans_not_experimental_evidence": {
                key: doc.get("full_text_spans", []) for survey in surveys
                for key, doc in survey.get("source_documents", {}).items()}}
        if idea_context:
            # Present only when discovery has a promising idea, so a prompt
            # recorded before discovery existed still replays byte for byte.
            context_blob["discovery_idea_to_compile"] = idea_context
        requirement = action.extra.get("experiment_requirement")
        if requirement:
            context_blob["experiment_requirement"] = requirement
            context_blob["compile_from_requirement_not_world_menu"] = True
        knowledge = _project_brief(kwargs) + (json.dumps(context_blob, ensure_ascii=False),)
        topic = str(state.get("goal") or "")
        common = dict(seed_label=topic, near_labels=(topic,), topic=topic,
                      context=tuple(f"{w.cite_id()}: {w.title}. {w.abstract}" for w in works),
                      knowledge=knowledge, program=(instructions, f"Research transformation: {operator}"),
                      research_mode=True)
        try:
            world = None
            if kwargs.get("world") is not None or kwargs.get("workspace") is not None or self.workspace is not None:
                try:
                    world = self._world(state, kwargs)
                except FileNotFoundError:
                    if kwargs.get("world") is not None:
                        raise
                    # An intent without a materialized freeze remains speculative.
            if world is not None and not requirement:
                from ..farcompile import menu_from_card
                menu = menu_from_card(None, world)
                # These are existing runtime registrations, not invented scout
                # responses, movement estimates, or evidence of effectiveness.
                # A claim-derived experiment requirement forbids the menu from
                # writing the question; it may only type-check later.
                if 'admitted_receipt.json' in world.files:
                    from .requirement import world_capability_view
                    api = world_capability_view(world)
                    common['knowledge'] += (json.dumps({'bound_world_api': api}, ensure_ascii=False),)
                    common['world_menu'] = replace(menu, levers=tuple(api['callable_operations']),
                        observables=tuple(api['metrics']), load_hint='', attested_fields=())
                else:
                    from ..dynworld import levers_of, observables_of
                    common["world_menu"] = replace(menu, levers=levers_of(world), observables=observables_of(world))
            if lane in ("assumption_reframe", "discovery_bridge"):
                card = generate_reframe(self.client, **common, operator=operator)
            else:
                labels = (operator.replace("_", " "),) if chosen else ("unconstrained mechanism exploration",)
                card = generate_card(self.client, **common, far_labels=labels, operator=operator,
                                     alienness=0.0, label_to_node={})
            if world is not None and card.claim_spec:
                card = replace(card, claim_spec={**card.claim_spec, "world_digest": world.digest})
            metadata = dict(card.research_metadata or {})
            required = ("assumptions", "competing_explanations", "closest_prior_work", "inherited_components",
                        "genuinely_changed_components", "claim_delta", "mechanism_delta", "why_not_reconstruction",
                        "experiment_proposal")
            if any(not metadata.get(k) for k in required) or card.falsifier == "claim_contains_a_falsifiable_assertion":
                raise ValueError("structured_hypothesis_contract_missing")
            citable = {w.cite_id() for w in works}
            relevance = next((r.get("retrieval_relevance") for r in reversed(surveys)
                              if r.get("retrieval_relevance")), None)
            if relevance and relevance.get("usable"):
                # A paper that merely shares a word with the question is not the
                # closest prior work to it.
                citable &= set(relevance["usable"])
            if not set(metadata["closest_prior_work"]).issubset(citable):
                raise ValueError("closest_prior_work_must_be_verified_literature_on_this_question")
            paper_targets = [{"target_state": p} for r in surveys for p in r.get("paper_states", [])]
            paper_targets += [{"target_state": w.to_dict()} for w in works]
            rejected = copy_check(card.to_dict(), transitions + paper_targets)
            if rejected:
                return CandidateResult(status="hypothesis_rejected", events=[make_event("EvidenceRejected", {
                    "theory_id": card.card_id, "epistemic": "GENERATED", "reasons": list(rejected),
                    "research_lane": lane, "question_id": action.target}, action_type=THEORIZE)],
                    extras={"reason": "; ".join(rejected)})
            alternatives = metadata["competing_explanations"]
            if not isinstance(alternatives, list) or any(not isinstance(a, dict) or not a.get("id") or not a.get("claim") for a in alternatives):
                raise ValueError("competing_explanations_require_structured_ids_and_claims")
            identifiers = [a["id"] for a in alternatives]
            proposal = dict(metadata["experiment_proposal"])
            compared = proposal.get("hypotheses_compared", [])
            outcomes = proposal.get("expected_outcomes", {})
            if (any(not isinstance(i, str) or i == "self" for i in identifiers)
                    or len(set(identifiers)) != len(identifiers)
                    or not isinstance(compared, list) or not isinstance(outcomes, dict)
                    or any(not isinstance(i, str) for i in compared)
                    or len(set(compared)) != len(compared)
                    or set(compared) != {"self", *identifiers} or set(outcomes) != set(compared)):
                raise ValueError("hypothesis_reference_contract: use unique competitor IDs and 'self'; "
                                 "hypotheses_compared and expected_outcomes must name exactly those IDs")
            id_map = {"self": card.card_id, **{a["id"]: f"{card.card_id}:alternative:{i}" for i, a in enumerate(alternatives)}}
            proposal["hypotheses_compared"] = [id_map[h] for h in proposal["hypotheses_compared"]]
            proposal["expected_outcomes"] = {id_map[h]: p for h, p in proposal["expected_outcomes"].items()}
            theory = {**metadata, "id": card.card_id, "question_id": action.target,
                "claim": card.claim, "text": card.claim, "mechanism": card.mechanism,
                "falsifier": card.falsifier, "discriminating_prediction": card.prediction,
                "predictions": [card.prediction], "card": card.to_dict(), "epistemic": "GENERATED",
                "research_lane": lane, "requested_lane": action.extra["research_lane"],
                "operator_reused": operator,
                "operator_selection": selection,
                "source_trajectories_used": [
                    {"source": t.source_cite_id, "target": t.target_cite_id, "operator": t.operator,
                     # How this trajectory was retrieved. An overlap match
                     # proposes a historical operator; it does not assert that
                     # the historical transition applies here.
                     "match_basis": (t.retrieval or {}).get("match_basis", "unknown"),
                     "applicability": (t.retrieval or {}).get("applicability", "unknown"),
                     "matched_fields": sorted((t.retrieval or {}).get("matched_fields") or {}),
                     } for t in chosen],
                "lane_fallback": lane_fallback,
                "trajectory_applicability": "established" if chosen and all(
                    (t.retrieval or {}).get("applicability") == "established" for t in chosen)
                    else ("proposed" if chosen else "none"),
                "hop": hop, "parent_theory_id": parent_id,
                "transition_kind": action.extra.get("transition_kind", "first_hop"),
                "revision_id": revision_id,
                # The discovery idea this registered hypothesis compiles. Its
                # evidence status is read back through this link, never written.
                "discovery_idea_id": bridge,
                "failure_pivots": int(action.extra.get("failure_pivots") or 0),
                "evidence_considered": [r["evidence_id"] for r in evidence],
                "evidence_context_digest": sha256_text(json.dumps(evidence, sort_keys=True)),
                "competing": [id_map[a["id"]] for a in alternatives], "experiment_proposal": proposal,
                "read_first": [w.cite_id() for w in works]}
            events = [make_event("TheoryCreated", theory, action_type=THEORIZE)]
            events += [make_event("TheoryCreated", {"id": id_map[a["id"]], "question_id": action.target,
                "claim": a["claim"], "text": a["claim"], "role": "competing_explanation",
                "competing": [card.card_id], "epistemic": "GENERATED"}, action_type=THEORIZE) for a in alternatives]
            from .reducer import apply_events
            from .state import ScientificState
            from .scheduler import hypothesis_resolution_priority
            preview = apply_events(ScientificState.from_dict(state), events)
            validity = hypothesis_resolution_priority(proposal, preview)
            if not validity["valid"]:
                raise ValueError(validity["reason"])
            return CandidateResult(status="theorized", events=events, extras={"theory": theory})
        except (GenerationRefused, LLMUnavailable, ValueError, KeyError, TypeError, OSError) as exc:
            return self._failure(action, exc, research_lane=lane)

    def probe(self, action, state, **kwargs):
        state = readonly_view(state)
        if str(getattr(action, "extra", {}).get("study_stage") or "") in {
                "pilot", "calibration", "scientific_pilot", "confirmatory", "replication"}:
            from types import SimpleNamespace

            from .executors import execute_computational_pilot

            workspace = kwargs.get("workspace") or self.workspace
            return execute_computational_pilot(
                SimpleNamespace(workspace=workspace, state=state), action)
        try:
            workspace = self._workspace(kwargs)
            world = self._world(state, kwargs)
            theory = self._theory(action, state)
            from .experiment import admitted_receipt, fixture_errors
            receipt = admitted_receipt(world)
            if receipt.get('admitted'):
                # An admitted action shares the world's scientific object
                # schema, not the legacy schema executor's data layout. Prose
                # cannot supply the missing experimental registration.
                from .requirement import EXPERIMENT, ExperimentRequirement, PILOT
                raw = action.extra.get('experiment_requirement')
                errors = []
                req = None
                if not isinstance(raw, Mapping) or not raw:
                    errors.append('missing_experiment_requirement')
                else:
                    try:
                        req = ExperimentRequirement.from_dict(raw)
                        errors.extend(fixture_errors(req, world))
                        if req.kind != EXPERIMENT or req.execution.get('operation') != receipt.get('action'):
                            errors.append('admitted_operation_not_registered')
                        if (req.claim_id != action.target or theory.get('id') != req.claim_id
                                or req.execution.get('hypothesis') != (theory.get('claim') or theory.get('text'))):
                            errors.append('registered_operational_theory_mismatch')
                    except (ValueError, TypeError, KeyError) as exc:
                        errors.append('invalid_experiment_requirement:' + str(exc))
                if action.extra.get('freeze') != world.id:
                    errors.append('missing_or_different_registered_world')
                if action.extra.get('study_stage'):
                    errors.append('invalid_study_stage')
                if errors:
                    return self._failure(action, 'admitted_probe_requires_registered_experiment',
                        status='probe_execution_blocked', gap='registered_experiment_definition',
                        scope='plan', registration_errors=errors, world_id=world.id,
                        required_operation=receipt.get('action'), required_metric=receipt.get('observable'))
                from types import SimpleNamespace
                from .executors import execute_computational_pilot
                registered_action = replace(action, extra={**action.extra, 'study_stage': PILOT})
                return execute_computational_pilot(
                    SimpleNamespace(workspace=workspace, state=state), registered_action)
            card = _card(action.extra.get("card") or theory.get("card"))
            if theory.get("card"):
                canonical = _card(theory["card"])
                if card.to_dict() != canonical.to_dict() or card.card_id != theory.get("id"):
                    raise ValueError("probe_card_does_not_match_canonical_theory")
            if theory.get("research_lane") and not any(t.get("id") == theory.get("id") for t in state.get("theories", [])):
                raise ValueError("probe_requires_canonical_theory")
            if str(card.idea_kind or "probe").lower() != "probe":
                raise ValueError("requires_probe_compilation_or_world_acquisition: "
                                 "a question/acquire/theory card is not an executable intervention")
            if self.client is None:
                raise ValueError("research_model_not_connected")
        except (OSError, ValueError, TypeError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked")
        topic = str(state.get("goal") or card.claim)
        works = self._works(state, str(theory.get("question_id") or ""))
        if theory.get("research_lane") and not any(a.get("kind") == "literature_survey" and
                a.get("theory_id") == theory.get("id") and a.get("works") for a in state.get("artifacts", [])):
            return self._failure(action, "hypothesis_specific_literature_check_required", status="probe_execution_blocked")
        prior = strongest_prior(card.claim, works) if works else None
        if prior and prior.kills:
            return self._failure(action, "claim_already_stated_in_verified_literature",
                                 status="probe_execution_blocked", prior=prior.to_dict())
        try:
            from ..claimspec import in_world_testable, scientific_for
            if not in_world_testable(card.claim_spec):
                return self._failure(action, "claim_not_testable_on_bound_world",
                                     status="probe_execution_blocked")
            claim_checked = False
            if card.claim_spec:
                if ((card.claim_spec.get("world_id") and card.claim_spec["world_id"] != world.id)
                        or (card.claim_spec.get("world_digest") and card.claim_spec["world_digest"] != world.digest)):
                    raise ValueError("claim_world_identity_mismatch")
                from ..claimspec import (ContractViolation, WorldObjectRegistry,
                                         claim_spec_from_payload, typecheck_claim)
                from ..dynworld import levers_of, observables_of
                try:
                    registered = claim_spec_from_payload(card.claim_spec)
                except ContractViolation as exc:
                    # A stored claim whose recorded identity contradicts itself
                    # fails object validation before there is anything to check.
                    return self._failure(action, "claim_object_validation_failed",
                                         status="probe_execution_blocked",
                                         claim_reasons=[str(exc)])
                checked = typecheck_claim(registered, WorldObjectRegistry.from_names(
                    (*levers_of(world), *observables_of(world))))
                claim_checked = bool(checked.ok and registered.claim_id == card.card_id
                                     and registered.statement == card.claim
                                     and registered.handle.id and registered.measurement.dv)
                if not claim_checked:
                    return self._failure(action, "claim_object_validation_failed",
                        status="probe_execution_blocked", claim_reasons=list(checked.reasons))
            scientific = scientific_for(card=card, world=world, workspace=workspace)
            diagnosis = write_diagnosis(self.client, card, topic, world=world, works=works,
                                       program=json.dumps({"experiment_proposal": theory.get("experiment_proposal"),
                                                           "competing_theories": theory.get("competing"),
                                                           "evidence_considered": theory.get("evidence_considered")}),
                                       scientific=scientific,
                                       prior_probe=action.extra.get("prior_probe"),
                                       failed_experiments=[str(row.get("experiment") or "")
                                                           for row in state.get("failed_designs", [])])
            diagnosis = replace(diagnosis, compute_tier=floor_tier(diagnosis.compute_tier, world.schema))
            if claim_checked and (diagnosis.handle_held or diagnosis.world_lever != registered.handle.id
                                  or diagnosis.world_observable != registered.measurement.dv):
                raise ValueError("diagnosis_differs_from_registered_claim")
            if diagnosis.world_lever == "none":
                return self._failure(action, "no_handle", status="probe_execution_blocked")
            spec = write_probe(self.client, card, diagnosis, topic, world=world,
                               scientific=scientific,
                               registered_measure=registered.measurement.dv if claim_checked else "",
                               prior_failure=action.extra.get("prior_failure"))
            if claim_checked and spec.measure.strip() != registered.measurement.dv:
                raise ValueError("probe_measure_differs_from_registered_claim")
            if not reads_world_data(spec.source, world):
                return self._failure(action, "probe_does_not_read_world", status="probe_policy_refused")
        except GenerationRefused as exc:
            return self._failure(action, exc.record.unlock_condition, status="probe_policy_refused",
                                 missing_capability=exc.record.missing_capability)
        except (LLMUnavailable, OSError, ValueError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked")
        return self._execute_registered(action, state, card, diagnosis, spec, theory, world,
                                        works, topic, workspace, claim_checked=claim_checked)

    def _execute_registered(self, action, state, card, diagnosis, spec, theory, world,
                            works, topic, workspace, *, claim_checked):
        """One immutable registered execution, shared by first and heldout probes."""
        experiment_digest = sha256_text(spec.source)
        eid = evidence_id(claim_id=card.card_id, experiment_digest=experiment_digest,
                          world_digest=world.digest, data_digest=world.digest)
        registration_workspace = workspace / "research_workers" / eid
        folder = registration_workspace / "candidates" / card.card_id
        if folder.exists():
            return self._failure(action, "registration_already_exists_use_verify",
                                 status="probe_execution_blocked", protocol_dir=str(folder))
        registration = {"kind": "WORLD", "source": spec.source, "measure": spec.measure,
                        "experiment_digest": experiment_digest, "world_digest": world.digest,
                        "data_digest": world.digest, "evidence_id": eid}
        brief = compile_brief(card, works, diagnosis=diagnosis, topic=topic)
        protocol = render_protocol(topic, card, brief, works, probe=registration, diagnosis=diagnosis, world=world)
        proposal = json.loads(json.dumps(theory.get("experiment_proposal") or {}))
        if not isinstance(proposal, dict):
            proposal = {}
        # These scientific decisions are committed before either arm runs.
        protocol.payload["experiment_proposal"] = proposal
        protocol.payload["diagnosis"] = diagnosis.to_dict()
        protocol.payload["card"] = card.to_dict()
        compared = proposal.get("hypotheses_compared") or []
        predictions = proposal.get("expected_outcomes") or {}
        if not isinstance(compared, list) or not all(isinstance(hid, str) for hid in compared):
            compared = []
        if not isinstance(predictions, Mapping):
            predictions = {}
        live_ids = {row.get("id") for row in state.get("theories", [])}
        competitors = [hid for hid in compared if hid != card.card_id]
        directions = {"increase": "treatment_higher", "decrease": "treatment_lower",
                      "treatment_higher": "treatment_higher", "treatment_lower": "treatment_lower"}
        own = predictions.get(card.card_id) or {}
        other = predictions.get(competitors[0]) or {} if len(competitors) == 1 else {}
        # The hypothesis proposal and the diagnosis were written at different
        # times against different vocabularies: a proposal may predict
        # "unchanged", which the diagnosis deliberately forbids because a
        # no-difference registration cannot compute a verdict. Requiring a
        # literal match between the two made the most common competing
        # explanation — "the effect is not there" — impossible to register, and
        # requiring the proposal to restate the observable identifier verbatim
        # meant two independently written strings had to be equal.
        #
        # What must hold is substantive: both hypotheses measured on the
        # registered observable, opposed predictions, and the proposal anchored
        # to the registration on at least one side.
        registered = str(diagnosis.world_observable or "").strip()
        declares = [str((row or {}).get("observable") or "")
                    for row in (own, other)]
        names_registered = bool(registered) and all(
            _names_observable(text, registered) for text in declares)
        own_arm = directions.get(own.get("direction"))
        other_arm = directions.get(other.get("direction"))
        anchored = (
            (own_arm is None or own_arm == diagnosis.expected_direction)
            and (other_arm is None or other_arm == diagnosis.alternative_direction)
            and (own_arm == diagnosis.expected_direction
                 or other_arm == diagnosis.alternative_direction))
        discriminator = bool(len(compared) == 2 and len(set(compared)) == 2
            and card.card_id in compared and set(compared) <= live_ids
            and isinstance(own, Mapping) and isinstance(other, Mapping)
            and names_registered
            and spec.measure.strip() == registered
            and own.get("direction") != other.get("direction")
            and anchored
            and diagnosis.expected_direction != diagnosis.alternative_direction
            and bool(proposal.get("minimal_discriminating_intervention"))
            and bool(proposal.get("nuisance_factors")))
        protocol.payload["discriminator_registered"] = discriminator
        # The registration is immutable: measured values live in probe.json.
        for key in ("treatment", "control", "verdict", "status"):
            (protocol.payload.get("cheap_probe") or {}).pop(key, None)
        try:
            folder = write_idea(registration_workspace, card_id=card.card_id, brief=brief.to_dict(),
                                works=[work.to_dict() for work in works], probe=registration,
                                protocol=protocol.payload, protocol_md=protocol.markdown, readme_md=protocol.readme)
            _json(folder / "card.json", card.to_dict())
            protocol_digest = sha256_text((folder / "protocol.json").read_text())
            tier = resolve_tier(diagnosis.compute_tier)
            dest = folder / "probe" / "execution"
            bind_world(dest, world)
            result = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
            if (digest_files(dest / "data", world.files) != world.digest
                    or sha256_text((folder / "experiment.py").read_text()) != experiment_digest
                    or sha256_text((folder / "protocol.json").read_text()) != protocol_digest):
                raise ValueError("registered source, protocol, or consumed world digest changed during execution")
        except (OSError, ValueError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked", failed=True,
                                 protocol_dir=str(folder))
        if result.status != "ran":
            _json(folder / "execution_failure.json", result.to_dict())
            return self._failure(action, result.error or result.status, status="probe_execution_blocked",
                                 failed=True, executor_status=result.status, diagnosis=diagnosis.to_dict(),
                                 protocol_dir=str(folder), experiment_digest=experiment_digest)
        measured = judge_probe(diagnosis, result.treatment, result.control)
        # Relief applies only when the diagnosis says which registered competitor
        # its alternative is. A discriminating registration against competitor C
        # establishes nothing about a prose alternative A, and the two are not
        # required to be the same thing — so an unstated link fails closed.
        linked = str(getattr(diagnosis, "alternative_hypothesis_id", "") or "")
        discriminated_alternative = linked if linked and linked in set(compared) else ""
        gate = self._gates(card, diagnosis, spec, result, world, folder, tier,
                           arithmetic=measured["verdict"],
                           discriminator=discriminator,
                           discriminated_alternative=discriminated_alternative)
        # Facts only: the arms, the registration, what each check measured and
        # whether it applied, the units counted. What they amount to is derived
        # by the validity contract — here for the local record, and again by the
        # kernel on admission and on every replay.
        evidence = {**result.to_dict(), **measured, **gate, **registration,
                    "executor_status": result.status, "role": "ProbeEvidence", "epistemic": "WORLD",
                    "execution_status": result.status,
                    "literature_checked": bool(works), "replication_required": True,
                    "probe_kind": "WORLD", "attested": True, "card_id": card.card_id,
                    "theory_id": theory.get("id") or card.card_id,
                    "question_id": theory.get("question_id") or action.frontier_target_id,
                    "world_id": world.id, "protocol_dir": str(folder), "protocol_digest": protocol_digest,
                    "experiment_proposal": proposal, "discriminator_registered": discriminator,
                    "diagnosis": diagnosis.to_dict(), "card": card.to_dict(),
                    "claim": card.claim, "harness_version": str(state.get("harness_version") or ""),
                    **_reported_sample(result.metrics, world)}
        if not card.claim_spec:
            evidence["scope_gap"] = "claim_spec_not_registered"
        limits = (state.get("contract") or {}).get("resource_limits")
        evidence.update(assess_probe(
            evidence, min_sample=sample_floor(limits if isinstance(limits, Mapping) else None),
        ).fields(evidence))
        _json(folder / "probe.json", evidence)
        _json(folder / "metrics.json", result.metrics)
        status = {"supports": "probe_positive", "weakens": "probe_scientifically_negative",
                  "uninformative": "probe_inconclusive"}[evidence["verdict"]]
        return CandidateResult(status=status, evidence=evidence, evidence_id=eid, world_id=world.id,
                               extras={"executor_status": result.status, "protocol_dir": str(folder)})

    def _gates(self, card, diagnosis, spec, result, world, folder, tier, *, arithmetic,
               discriminator=False, discriminated_alternative=""):
        """Fixed validation work, never a redesign loop or sign-seeking retry.

        Runs the checks this registration calls for and records what they
        measured — the oracle placebo's arms, one separation per placebo world —
        and whether each applied. It does not decide what they mean; the
        validity contract does, from these records.
        """
        from ..dvsanity import materialize_oracle_placebo, oracle_dependent, oracle_field
        from ..dynworld import has_dynamics, levers_of, materialize_placebo
        from ..evidence import ablation_reason
        from ..validity import DIRECTIONAL, MIN_PLACEBO_RUNS, PASS, placebo_check, separation

        reason = ablation_reason(diagnosis.alternative, diagnosis.experiment,
                                 levers=levers_of(world), world_lever=diagnosis.world_lever)
        gate = {"validation_gaps": [], "ablation_required": bool(reason),
                "ablation_reason": reason}
        # An alternative that names another lever of this world is an unshut
        # handle, and only shutting it will do. An alternative that names no
        # lever means the mechanism is not identified *by lever vocabulary* —
        # and a pre-registered comparison on the same measured observable, with
        # opposed predictions the verdict is computed from, identifies it
        # another way. Without this distinction every design in a world with
        # levers was permanently unidentified, and no result could corroborate.
        if (reason == "alternative_names_no_lever_of_this_world" and discriminator
                and discriminated_alternative):
            gate.update(ablation_required=False,
                        mechanism_identified="registered_discriminating_comparison",
                        discriminated_alternative=discriminated_alternative)
        elif reason:
            gate.update(ablation_required=True, mechanism_identified=False)
            gate["missing_capability"] = "registered_competing_mechanism_ablation"
        try:
            # Whether each check applies is a fact about this claim and world,
            # fixed before any check runs. A check that applied and then did not
            # run is an undefined result, not an absent flag.
            gate["oracle_required"] = bool(oracle_field(world.schema) and oracle_dependent(
                card.claim, diagnosis.experiment, diagnosis.treatment_arm, diagnosis.control_arm))
            gate["placebo_required"] = bool(has_dynamics(world))
            world_payload = world.root / "world.json"
            if world_payload.is_file():
                scope = json.loads(world_payload.read_text()).get("research_scope")
                if scope == "observational_trace_analysis_only":
                    gate["scope_ceiling"] = scope
                    gate["validation_gaps"].append("observational_archive_not_live_harness_execution")
            if gate["oracle_required"]:
                fixture = materialize_oracle_placebo(world, folder / "checks" / "oracle")
                if fixture is None:
                    gate["validation_gaps"].append("oracle_sanity_not_identifiable")
                else:
                    dest = folder / "checks" / "oracle-run"
                    bind_world(dest, fixture)
                    check = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
                    if check.status != "ran":
                        gate["validation_gaps"].append("oracle_sanity_execution_failed")
                    else:
                        gate["oracle_placebo"] = {"treatment": check.treatment, "control": check.control}
            # Every directional reading is set against placebo worlds, not only
            # a positive one: a negative separation the script also produces on
            # a ruined copy says as little about the hypothesis as a positive one.
            if gate["placebo_required"] and arithmetic in DIRECTIONAL:
                separations = []
                for seed in range(MIN_PLACEBO_RUNS):
                    fixture = materialize_placebo(world, folder / "checks" / f"placebo-{seed}", seed=seed)
                    dest = folder / "checks" / f"placebo-run-{seed}"
                    bind_world(dest, fixture)
                    check = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
                    if check.status != "ran":
                        raise ValueError(f"world consumption check {check.status}")
                    separations.append(separation(check.treatment, check.control))
                gate["placebo_separations"] = separations
                reading = placebo_check(separation(result.treatment, result.control),
                                        separations, diagnosis.margin)
                chain.append_event(folder / "chain.jsonl", chain.EXECUTE_PLACEBO,
                    {"card_id": card.card_id, "placebo_separations": separations,
                     "consumed": reading.status == PASS, "reading": reading.reason,
                     "experiment_digest": sha256_text(spec.source), "world_digest": world.digest})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            gate["validation_gaps"].append(str(exc))
        return gate

    def verify(self, action, state, **kwargs):
        state = readonly_view(state)
        target = next((row for row in state.get("evidence_records", [])
                       if row.get("evidence_id") == action.target), None)
        if target is None:
            return self._failure(action, "no_target_probe_evidence")
        try:
            workspace = self._workspace(kwargs)
            original_world_id = str(target.get("world_id") or "")
            if not original_world_id or Path(original_world_id).name != original_world_id or original_world_id in {".", ".."}:
                raise ValueError("invalid original evidence world id")
            world = load_fixture(workspace / "worlds" / original_world_id)
            if not is_attested(world):
                raise ValueError("original evidence world cannot be attested")
            if target.get("world_id") != world.id or target.get("world_digest") != world.digest:
                raise ValueError("target evidence belongs to another world; transfer is unverified")
            folder = Path(target["protocol_dir"])
            if not folder.resolve().is_relative_to(self._workspace(kwargs).resolve()):
                raise ValueError("protocol is outside the research workspace")
            protocol_text = (folder / "protocol.json").read_text()
            if sha256_text(protocol_text) != target.get("protocol_digest"):
                raise ValueError("registered protocol was changed")
            protocol = json.loads(protocol_text)
            for key in ("experiment_digest", "world_digest", "data_digest", "evidence_id"):
                if not target.get(key) or protocol.get(key) != target[key]:
                    raise ValueError(f"registered {key} differs from the target evidence")
            source = (folder / "experiment.py").read_text()
            if sha256_text(source) != target["experiment_digest"]:
                raise ValueError("registered experiment source was changed")
            replication_id = str(action.extra.get("replication_world_id") or "")
            if "replication_world_id" not in action.extra and target.get("reproduction_ok"):
                replication_id = next((str(row.get("id") or row.get("world_id") or "")
                    for row in state.get("world_versions", []) if row.get("role") == "heldout"
                    and not row.get("deprecated") and (row.get("id") or row.get("world_id")) != world.id), "")
            if replication_id:
                if Path(replication_id).name != replication_id or replication_id in {".", ".."}:
                    raise ValueError("invalid heldout world id")
                heldout = load_fixture(workspace / "worlds" / replication_id)
                if not is_attested(heldout):
                    raise ValueError("heldout world must already be attested")
                if (heldout.id == world.id or heldout.digest == world.digest
                        or (heldout.source_digest and heldout.source_digest == world.source_digest
                            and heldout.slice_rule == world.slice_rule)):
                    raise ValueError("heldout replication requires different attested scientific data")
                if heldout.schema != world.schema:
                    raise ValueError("heldout world has a different scientific object schema")
                if world.domains and heldout.domains and not set(world.domains).intersection(heldout.domains):
                    raise ValueError("heldout world belongs to a different scientific domain")
                if protocol.get("card") != target.get("card") or protocol.get("diagnosis") != target.get("diagnosis"):
                    raise ValueError("complete claim and diagnosis must match the immutable registration")
                card = _card(protocol["card"])
                diagnosis = diagnosis_from_payload(card.card_id, protocol["diagnosis"])
                from ..claimspec import WorldObjectRegistry, claim_spec_from_payload, typecheck_claim
                from ..dynworld import levers_of, observables_of
                registered = claim_spec_from_payload(card.claim_spec)
                checked = typecheck_claim(registered, WorldObjectRegistry.from_names(
                    (*levers_of(heldout), *observables_of(heldout))))
                if (not checked.ok or registered.claim_id != card.card_id or registered.statement != card.claim
                        or registered.handle.id != diagnosis.world_lever
                        or registered.measurement.dv != diagnosis.world_observable):
                    raise ValueError("registered claim is not admissible on the heldout scientific object")
                if floor_tier(diagnosis.compute_tier, heldout.schema) != diagnosis.compute_tier:
                    raise ValueError("heldout world exceeds the pre-registered compute tier")
                from ..probeexp import spec_from_payload
                spec = spec_from_payload({"measure": str((protocol.get("cheap_probe") or {}).get("measure") or ""),
                    "source": source}, tier=resolve_tier(diagnosis.compute_tier), claim=card.claim,
                    mechanism=card.mechanism, topic=str(state.get("goal") or ""), schema=heldout.schema)
                if not reads_world_data(source, heldout):
                    raise ValueError("registered script does not read heldout world payload")
                theory = {"id": target.get("theory_id") or card.card_id,
                          "question_id": target.get("question_id"),
                          "experiment_proposal": protocol.get("experiment_proposal") or {}}
                works = [FreshWork.from_dict(row) for row in json.loads((folder / "papers.json").read_text())]
                execution = self._execute_registered(action, state, card, diagnosis, spec, theory, heldout,
                    works, str(protocol.get("topic") or state.get("goal") or ""), workspace, claim_checked=True)
                if execution.evidence is None:
                    return self._failure(action, execution.extras.get("reason") or execution.status,
                        transfer_ok=False, replication_world_id=heldout.id)
                replicated = dict(execution.evidence)
                new_folder = Path(replicated["protocol_dir"])
                host = execute_protocol(new_folder, catalog_root=kwargs.get("catalog_root") or workspace,
                                        prefer_parent=False)
                identities_match = all(host.get(key) == replicated.get(key) for key in
                    ("experiment_digest", "world_digest", "data_digest", "evidence_id"))
                reproduced = bool(host.get("ok") and identities_match and
                    host.get("treatment") == replicated.get("treatment") and
                    host.get("control") == replicated.get("control"))
                supports_transfer = bool(reproduced and target.get("outcome") == "supports"
                    and replicated.get("outcome") == "supports" and replicated.get("claim_consistent")
                    and not replicated.get("cannot_corroborate"))
                independent = bool(world.source_digest and heldout.source_digest
                                   and world.source_digest != heldout.source_digest)
                replicated.update(replication_of=target["evidence_id"], verify_count=1,
                    verify_kind="VERIFY_REPLICATION", verified_by="hostexp.execute_protocol",
                    reproduction_ok=reproduced, replay_ok=reproduced, identity_ok=identities_match,
                    executable_ok=bool(host.get("ok")), transfer_ok=supports_transfer,
                    cross_world_transfer=supports_transfer, independent_replication=independent and reproduced,
                    replication="E3_independent" if independent else "E2_scale_replication",
                    world_compatible=True, host_record=host,
                    # Where this result is known to hold. Read by the archive
                    # projection to decide what is still relevant after the
                    # controller binds a different world; nothing wrote it
                    # before, so that projection always fell back.
                    valid_in_worlds=([world.id, heldout.id] if supports_transfer
                                     else [heldout.id]))
                _json(new_folder / "probe.json", replicated)
                return CandidateResult(status="replicated" if supports_transfer else "replication_inconclusive",
                    events=[make_event("EvidenceAdded", replicated, action_type=VERIFY,
                                       frontier_target_id=action.frontier_target_id)],
                    evidence_id=replicated["evidence_id"], world_id=heldout.id, unlocked=supports_transfer,
                    extras={"replication_of": target["evidence_id"], "transfer_ok": supports_transfer,
                            "reproduction_ok": reproduced, "protocol_dir": str(new_folder)})
            record = execute_protocol(folder, catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs),
                                      prefer_parent=False)
            same = all(record.get(key) == target.get(key) for key in
                       ("experiment_digest", "world_digest", "data_digest", "evidence_id"))
            replay_ok = bool(record.get("ok") and same and
                             record.get("treatment") == target.get("treatment") and
                             record.get("control") == target.get("control"))
            updated = {**target, "verify_count": int(target.get("verify_count") or 0) + 1,
                       "verify_kind": str(action.extra.get("verify_kind") or "VERIFY_REPLICATION"),
                       "verified_by": "hostexp.execute_protocol", "world_compatible": True,
                       "replay_ok": replay_ok, "replication": record.get("replication"),
                       "cross_world_transfer": False, "independent_replication": False,
                       "reproduction_ok": replay_ok, "identity_ok": same,
                       "executable_ok": bool(record.get("ok")),
                       "transfer_ok": False, "replication_required": True,
                       "host_record": record}
            if action.extra.get("diagnostic"):
                from .discovery import diagnose_observation
                updated["diagnosed"] = replay_ok
                updated["failure_diagnosis"] = diagnose_observation(updated)
            return CandidateResult(status="verified" if replay_ok else "verify_failed",
                events=[make_event("EvidenceAdded", updated, action_type=VERIFY,
                                   frontier_target_id=action.frontier_target_id)],
                evidence_id=str(target["evidence_id"]), world_id=world.id, unlocked=replay_ok,
                extras={"replay_ok": replay_ok, "same_instance": same, "independent_replication": False,
                        "cross_world_transfer": False, "host_record": record})
        except (ExecuteError, GenerationRefused, OSError, ValueError, KeyError, TypeError) as exc:
            return self._failure(action, exc, same_instance=False, cross_world_transfer=False)

    def synthesize(self, action, state, **kwargs):
        state = readonly_view(state)
        related = [row for row in state.get('evidence_records', [])
                   if not action.target or action.target in {
                       row.get('theory_id'), row.get('evidence_id'), row.get('question_id')}]
        computational = [row for row in related
                         if row.get('role') == 'ProbeEvidence' and row.get('attested')
                         and not row.get('invalidated') and row.get('execution_status') == 'ran'
                         and (row.get('registration') or {}).get('version') in {
                             'computational-protocol-v2', 'computational-protocol-v3'}]
        if computational:
            return self._synthesize_computational(action, state, computational, **kwargs)
        rows = [row for row in related
                if row.get("role") == "ProbeEvidence" and row.get("executor_status") == "ran"
                and row.get("attested") and not row.get('invalidated') and row.get("protocol_dir")]
        if not rows:
            return self._failure(action, "no_executed_research_evidence")
        selected = next((row for row in rows if action.target in
                         {row.get("theory_id"), row.get("evidence_id"), row.get("question_id")}), rows[-1])
        try:
            folder = Path(selected["protocol_dir"])
            if not folder.resolve().is_relative_to(self._workspace(kwargs).resolve()):
                raise ValueError("protocol is outside the research workspace")
            protocol_text = (folder / "protocol.json").read_text()
            if sha256_text(protocol_text) != selected.get("protocol_digest"):
                raise ValueError("registered protocol was changed")
            protocol = json.loads(protocol_text)
            registered_probe = json.loads((folder / "probe.json").read_text())
            for key in ("evidence_id", "experiment_digest", "world_digest", "data_digest", "treatment", "control"):
                if registered_probe.get(key) != selected.get(key):
                    raise ValueError(f"synthesis evidence differs from registered probe: {key}")
            if protocol.get("card") and protocol["card"] != selected.get("card"):
                raise ValueError("synthesis card differs from immutable registration")
            card = _card(protocol.get("card") or selected["card"])
            diagnosis = diagnosis_from_payload(card.card_id, protocol.get("diagnosis") or selected["diagnosis"])
            works = [FreshWork.from_dict(row) for row in json.loads((folder / "papers.json").read_text())]
            brief = compile_brief(card, works, diagnosis=diagnosis, topic=str(state.get("goal") or ""))
            from ..paperplan import gather_facts
            facts = gather_facts(folder.parent.parent, folder, card.card_id,
                                 catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs))
            scope = facts["scientific_scope"]
            from ..writeup import render_note
            # The legacy note's probe paragraph assumes constructed data. Supply
            # its measured paragraph from the same registered facts as the draft.
            note = render_note(str(state.get("goal") or ""), card, brief, works, probe=None)
            measurement = (
                f"Registered comparison on world `{facts['world'].get('id')}` "
                f"(digest `{scope['world_digest']}`), EvidenceID `{scope['evidence_id']}`: "
                f"treatment = {facts['probe'].get('treatment')}, control = {facts['probe'].get('control')}; "
                f"arithmetic verdict = `{scope['arithmetic_verdict']}`. "
                f"Claim strength: `{scope['claim_strength']}`. An arithmetic result does not "
                "establish a mechanism or generalization beyond the registered scope."
            )
            markdown = note.markdown.replace("No probe was run.", measurement, 1)
            markdown = markdown.replace("## Claim\n", "## Claim (GENERATED hypothesis)\n", 1)
            path = folder / "research_note.md"
            path.write_text(markdown + "\n\nScientific scope: " +
                            json.dumps(scope, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            _json(folder / "PAPER_FACTS.json", facts)
            artifact = {"id": f"synthesis-{selected['evidence_id']}", "kind": "research_synthesis",
                        "path": str(path), "evidence_id": selected["evidence_id"],
                        "question_id": selected.get("question_id"), "verdict": selected.get("verdict"),
                        "cannot_corroborate": scope["cannot_corroborate"],
                        "read_first": list(brief.read_first), "protocol_dir": str(folder)}
            artifact["facts_path"] = str(folder / "PAPER_FACTS.json")
            if action.extra.get("paper_draft"):
                if self.client is None:
                    return self._failure(action, "paper_draft_requires_research_model")
                from ..paperplan import compile_paper
                artifact["paper"] = compile_paper(folder.parent.parent, folder, card.card_id,
                    client=self.client, rounds=0, lang=str(action.extra.get("lang") or "en"),
                    catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs))
            return CandidateResult(status="synthesized", events=[make_event("ArtifactAdded", artifact,
                action_type=SYNTHESIZE, frontier_target_id=action.frontier_target_id)], extras=artifact)
        except (OSError, ValueError, KeyError, TypeError, LLMUnavailable) as exc:
            return self._failure(action, exc)

    def _synthesize_computational(self, action, state, rows, **kwargs):
        """Report admitted observations, without supplying another scientific verdict."""
        from .experiment import admission_errors, confined_path, digest
        from .journal import research_journal
        from .state import ScientificState
        try:
            workspace = self._workspace(kwargs)
            scientific = ScientificState.from_dict(dict(state))
            for row in rows:
                errors = admission_errors(row, workspace, scientific)
                if errors:
                    raise ValueError(f"synthesis_receipt_invalid:{row['evidence_id']}: {'; '.join(errors)}")
            journal = research_journal(scientific)
            facts = {
                'source': 'admitted_computational_receipts',
                'observations': [{key: row.get(key) for key in (
                    'evidence_id', 'question_id', 'theory_id', 'study_stage', 'world_id', 'world_digest',
                    'registration', 'registration_digest', 'treatment', 'control', 'primary_metric',
                    'sample_size', 'sample_size_basis', 'outcome', 'validity', 'inference_link',
                    'cannot_corroborate', 'protocol_scope', 'replication_of', 'independent_replication')}
                    for row in rows],
                'beliefs': scientific.belief_states,
                'current_questions': journal['current_questions'],
                'remaining_critical_gaps': journal['remaining_critical_gaps'],
                'refutation_and_invalidation_history': journal['refutation_and_invalidation_history'],
                'mission_completion': (scientific.quality or {}).get('mission_completion', {}),
                'scope': 'Each observation retains its frozen protocol and world scope; '
                         'calibration and pilot results do not confirm a scientific hypothesis. '
                         'This report adds no evidence and establishes no broader mechanism or generalization.',
            }
            identity = digest(facts)[:24]
            folder = confined_path(workspace, 'research_workers', f'synthesis-{identity}')
            facts_path = folder / 'PAPER_FACTS.json'
            _json(facts_path, facts)
            path = folder / 'research_note.md'
            path.write_text('# Research evidence synthesis\n\n' + facts['scope'] +
                '\n\n```json\n' + json.dumps(facts, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
            artifact = {'id': f'synthesis-{identity}', 'kind': 'research_synthesis',
                        'path': str(path), 'facts_path': str(facts_path),
                        'evidence_id': rows[-1]['evidence_id'],
                        'evidence_ids': [row['evidence_id'] for row in rows],
                        'question_id': rows[-1].get('question_id'),
                        'epistemic': 'GENERATED', 'cannot_corroborate': True,
                        'document_scope': 'evidence_synthesis'}
            return CandidateResult(status='synthesized', events=[make_event(
                'ArtifactAdded', artifact, action_type=SYNTHESIZE,
                frontier_target_id=action.frontier_target_id)], extras=artifact)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return self._failure(action, exc)
