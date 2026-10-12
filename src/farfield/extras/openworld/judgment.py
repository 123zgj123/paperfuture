"""Who judges novelty and importance, and what that judgment is allowed to do.

No execution settles whether a result is new or whether it matters. Until this
module the two dimensions had one producer between them — a lexical prior-art
check that can say "nothing retrieved restates this claim" — and importance had
none at all: across sixty-four live workspaces it read ``unknown`` in every one,
so any contract that requires it was unreachable by construction.

A panel of model roles can supply the judgment. It is a *worker behind the
existing REVIEW action*, not a second controller: it proposes ``ReviewRecorded``
rows, trusted admission checks what they claim to have judged, and the derived
quality view does the grading. Nothing here writes to state, adds evidence,
raises belief or answers a question.

What several roles buy, and what they do not:

* An examiner, an advocate and a skeptic see different instructions and do not
  see each other, so one framing cannot carry all three. An adjudicator reads
  the two cases unlabelled, in an order fixed by the subject rather than by who
  wrote them.
* Roles served by one model are still one model. Their agreement is correlated
  and is recorded as ``same_model_roles``; a caller may bind any role to a
  different model and the record then says ``distinct_models``. Either way
  agreement is not evidence, and no number of judges reaches ``adequate``
  without a citation this state actually verified.
* Grades are combined by code, downwards: novelty is the lower of the examiner
  and the adjudicator, importance cannot exceed what its own advocate claimed,
  an examiner who skipped the nearest prior work the code selected is capped,
  and a literature pool that never reaches the claim cannot certify novelty.
  The skeptic's grade and the unresolved disagreement are kept, not averaged.

A judgment is about a *subject*: the claims that were finally measured, how each
came out, and the literature they were checked against. Change any of those and
the judgment is stale. The same subject is never judged twice, so a panel cannot
be re-rolled for a better grade.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PANEL_VERSION = "judgment-panel-v1"
SOURCE = "judgment_panel"

EXAMINER, ADVOCATE, SKEPTIC, ADJUDICATOR = "examiner", "advocate", "skeptic", "adjudicator"
ROLES = (EXAMINER, ADVOCATE, SKEPTIC, ADJUDICATOR)

#: How many of the retrieved works closest to the claim the examiner must address.
NEAREST_PRIOR = 3

#: How many times one project epoch may send a judged result back for revision.
REVISION_CEILING = 2

#: The operator that answers each shortfall. A known result needs a frame the
#: local literature does not already hold; an unimportant one needs its problem
#: restated so that the valuable phenomenon is the one being claimed.
REVISION_OPERATORS = {"novelty": "REMOTE_ESCAPE", "importance": "REFORMULATE"}

#: The discovery mode each operator runs under. Only REMOTE_ESCAPE actually
#: queries the remote concept search; TRANSFER_STRUCTURE under EVOLVE asks the
#: model to borrow from another field and hands it nothing from one.
REVISION_MODES = {"REMOTE_ESCAPE": "REMOTE_ESCAPE", "REFORMULATE": "EVOLVE"}

_GRADES = "one of missing, weak, partial, adequate"

_COMMON = (
    "You are one member of a review panel inside an automated research runtime. The "
    "measurements you are shown were admitted by the runtime and are not yours to "
    "re-litigate; whether the claim is true is already decided by them. You judge only "
    "what no measurement can: whether the finding is new and whether it matters. "
    "Cite literature only by the cite_id values you are given — any other id is "
    "discarded. Grades are " + _GRADES + ". Reply with one JSON object and nothing else. ")

SYSTEM = {
    EXAMINER: _COMMON + (
        "Your role: prior-art examiner. For EVERY work listed under nearest_prior_works, "
        "say what it shares with the finding and what differs. Then say what, if anything, "
        "is new. If the works you were given do not reach this claim at all, say so and list "
        "the searches that would. Schema: {\"closest_prior\": [{\"cite_id\", \"overlap\", "
        "\"difference\"}], \"what_is_new\": str, \"novelty\": grade, \"basis\": str, "
        "\"prior_art_search_needed\": [str]}"),
    ADVOCATE: _COMMON + (
        "Your role: make the strongest HONEST case that this finding is new and that it "
        "matters — who would do what differently because of it. Do not inflate; a case you "
        "cannot make should be graded low by you. Schema: {\"case\": str, "
        "\"who_changes_what\": str, \"novelty\": grade, \"importance\": grade, "
        "\"refs\": [cite_id]}"),
    SKEPTIC: _COMMON + (
        "Your role: make the strongest HONEST case that this finding is already known, "
        "follows from definitions or arithmetic, or changes nothing for anyone. Ask whether "
        "a domain expert would have predicted the reading without running it. Schema: "
        "{\"case\": str, \"expert_would_predict_without_running\": bool, "
        "\"novelty\": grade, \"importance\": grade, \"refs\": [cite_id]}"),
    ADJUDICATOR: _COMMON + (
        "Your role: adjudicator. You are given a prior-art table and two unlabelled cases "
        "that were written independently. Decide which arguments survive. Do not split the "
        "difference: name the decisive argument, and state any disagreement you could not "
        "resolve. Say concretely what would have to change for the work to be new, and for "
        "it to matter. Schema: {\"novelty\": grade, \"importance\": grade, "
        "\"decisive_argument\": str, \"unresolved_disagreement\": str, "
        "\"what_would_make_it_new\": str, \"what_would_make_it_matter\": str, "
        "\"novelty_refs\": [cite_id], \"importance_refs\": [cite_id]}"),
}


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _rank() -> Mapping[str, int]:
    from .quality import STATUS_RANK

    return STATUS_RANK


def _grade(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"missing", "weak", "partial", "adequate"} else ""


def _lowest(*grades: str) -> str:
    rank = _rank()
    known = [grade for grade in grades if grade]
    return min(known, key=lambda grade: rank[grade]) if known else ""


def _cap(grade: str, ceiling: str) -> str:
    from .quality import cap

    return cap(grade, ceiling) if grade else grade


# ---------------------------------------------------------------------------
# What is being judged
# ---------------------------------------------------------------------------


def _verified_works(state: Any) -> dict[str, dict[str, str]]:
    """Retrieved works a judgment may cite — the same set ``quality`` grounds on."""
    from ..livefeed import FreshWork

    surveys = [row for row in getattr(state, "artifacts", []) or []
               if row.get("kind") == "literature_survey"]
    citable = {str(cite) for row in surveys
               for cite in list(row.get("citations") or []) + list(row.get("read_first") or [])}
    relevance = {str(paper.get("cite_id") or ""): str(paper.get("relevance") or "")
                 for survey in surveys
                 for paper in (survey.get("retrieval_relevance") or {}).get("papers") or []}
    works: dict[str, dict[str, str]] = {}
    for survey in surveys:
        for raw in survey.get("works") or []:
            try:
                work = FreshWork.from_dict(raw)
            except (TypeError, ValueError, KeyError):
                continue
            cite = work.cite_id()
            if cite and cite in citable:
                works.setdefault(cite, {"cite_id": cite, "title": work.title,
                                        "abstract": str(work.abstract or "")[:700],
                                        "relevance": relevance.get(cite, "")})
    return works


def _findings(state: Any) -> list[dict[str, Any]]:
    from .quality import result_review_scope

    ids = set(result_review_scope(state)["evidence_ids"])
    ideas = {str(row.get("id") or ""): dict(row.get("idea") or {})
             for row in getattr(state, "artifacts", []) or []
             if row.get("kind") == "discovery_idea"}
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in getattr(state, "evidence_records", []) or []:
        if str(row.get("evidence_id") or "") not in ids:
            continue
        # A reading that came back uninformative changes nothing that was
        # judged. A live panel's verdict was unseated two ticks later by exactly
        # such a reading from an unrelated lane.
        if str(row.get("outcome") or "") not in {"supports", "weakens"}:
            continue
        idea_id = str(row.get("discovery_idea_id") or row.get("theory_id") or "")
        execution = dict((row.get("registration") or {}).get("execution") or {})
        claim = str(execution.get("hypothesis") or ideas.get(idea_id, {}).get("central_claim") or "")
        outcome = str(row.get("outcome") or "")
        found.setdefault((idea_id, claim, outcome), {
            "idea_id": idea_id, "claim": claim, "outcome": outcome,
            "metric": str(row.get("primary_metric") or ""),
            "treatment": row.get("treatment"), "control": row.get("control"),
            "scope": str(row.get("protocol_scope") or ""),
            "prohibited_inferences": list(row.get("prohibited_inferences") or [])[:6],
        })
    return [found[key] for key in sorted(found)]


def judgment_subject(state: Any) -> dict[str, Any]:
    """The thing a panel would judge, with an identity no wording can change.

    Empty until a final reading exists: before that there is a proposal, and the
    critic already speaks to proposals.
    """
    findings = _findings(state)
    if not findings:
        return {}
    from .contract import contract_from_state

    works = _verified_works(state)
    mission = str(contract_from_state(state).root_question or getattr(state, "goal", "") or "")
    identity = {
        "version": PANEL_VERSION, "mission": mission,
        "findings": [{key: row[key] for key in ("idea_id", "claim", "outcome")}
                     for row in findings],
        "literature": sorted(works),
        "epoch": int(getattr(state, "project_epoch", 0) or 0),
    }
    return {"digest": _digest(identity), "mission": mission, "findings": findings,
            "literature": sorted(works)}


def _panel_rows(state: Any, digest: str) -> list[dict[str, Any]]:
    return [dict(row) for row in getattr(state, "reviews", []) or []
            if row.get("source") == SOURCE
            and str((row.get("panel") or {}).get("subject_digest") or "") == digest]


def required_judgments(state: Any) -> dict[str, str]:
    from .contract import contract_from_state
    from .quality import JUDGMENT_DIMENSIONS, requirements_for

    requirements = requirements_for(contract_from_state(state))
    return {name: requirements[name] for name in sorted(JUDGMENT_DIMENSIONS)
            if name in requirements}


def judgment_due(state: Any) -> list[str]:
    """Required judgment dimensions nobody has judged for the current subject."""
    required = required_judgments(state)
    if not required:
        return []
    subject = judgment_subject(state)
    if not subject:
        return []
    judged = {str(row.get("dimension") or "") for row in _panel_rows(state, subject["digest"])}
    return [name for name in required if name not in judged]


def judged_short(state: Any) -> dict[str, dict[str, str]]:
    """Required dimensions the panel graded below the contract, for the current subject."""
    from .quality import meets

    subject = judgment_subject(state)
    if not subject:
        return {}
    required = required_judgments(state)
    short: dict[str, dict[str, str]] = {}
    for row in _panel_rows(state, subject["digest"]):
        name = str(row.get("dimension") or "")
        if name in required and not meets(str(row.get("assessment") or ""), required[name]):
            panel = dict(row.get("panel") or {})
            short[name] = {"judged": str(row.get("assessment") or ""), "required": required[name],
                           "decisive_argument": str(row.get("assessment_basis") or ""),
                           "what_would_raise_it": str(panel.get("improve") or ""),
                           "independence": str(panel.get("independence") or ""),
                           "transcript": str(panel.get("transcript") or "")}
    return short


def admission_error(payload: Mapping[str, Any], state: Any) -> str:
    """A panel row may only speak about the subject this state actually holds."""
    if payload.get("source") != SOURCE and not isinstance(payload.get("panel"), Mapping):
        return ""
    claimed = str((payload.get("panel") or {}).get("subject_digest") or "")
    current = str(judgment_subject(state).get("digest") or "")
    if not current or claimed != current:
        return "judgment_panel_subject_is_not_the_current_subject"
    return ""


# ---------------------------------------------------------------------------
# A low grade is a reason to improve the idea, a bounded number of times
# ---------------------------------------------------------------------------


def revision_request(state: Any) -> dict[str, Any]:
    """What the panel said would have to change, if the result fell short.

    The shortfall named first is the one addressed; a revised idea produces a new
    subject and is judged again. Bounded per project epoch, because a panel that
    is never satisfied is otherwise an unbounded spend.
    """
    subject = judgment_subject(state)
    if not subject:
        return {}
    from .quality import meets

    required = required_judgments(state)
    rows = {str(row.get("dimension") or ""): row for row in _panel_rows(state, subject["digest"])}
    epoch = int(getattr(state, "project_epoch", 0) or 0)
    # A revision is spent when it produced an idea. One refused before any model
    # was called revised nothing: the second live build charged two such
    # refusals to the ceiling and ended having changed no idea at all.
    spent = sum(1 for row in getattr(state, "artifacts", []) or []
                if row.get("kind") == "action_attempt" and row.get("judged_below_the_bar")
                and int(row.get("project_epoch") or 0) == epoch
                and int(row.get("events_admitted") or 0) > 0)
    if spent >= REVISION_CEILING:
        return {}
    for name in ("importance", "novelty"):
        row = rows.get(name)
        if not row or name not in required or meets(str(row.get("assessment") or ""), required[name]):
            continue
        panel = dict(row.get("panel") or {})
        wanted = str(panel.get("improve") or "").strip()
        idea_ids = [item["idea_id"] for item in subject["findings"] if item.get("idea_id")]
        return {"dimension": name, "operator": REVISION_OPERATORS[name],
                "mode": REVISION_MODES[REVISION_OPERATORS[name]],
                "decisive_argument": str(row.get("assessment_basis") or ""),
                "what_would_raise_it": wanted,
                "must_change": f"{name}: {wanted}" if wanted else name,
                "review_id": str(row.get("review_id") or ""),
                "idea_ids": idea_ids, "judged": str(row.get("assessment") or ""),
                "required": required[name], "revisions_spent": spent}
    return {}


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def _screen(claims: list[str], works: Mapping[str, dict[str, str]]) -> None:
    """Fill in a subject screen for works whose survey recorded none.

    The survey screens against the mission. Where it did not, the same lexical
    anchor guard is applied against the claims — one rule, not a second one.
    """
    from .exploration import paper_relevance

    order = {"on_topic": 2, "adjacent": 1, "off_topic": 0, "unknown": 1}
    for work in works.values():
        if work.get("relevance"):
            continue
        verdicts = [str(paper_relevance(claim, work).get("relevance") or "unknown")
                    for claim in claims] or ["unknown"]
        work["relevance"] = max(verdicts, key=lambda verdict: order.get(verdict, 1))


def _nearest_prior(claims: list[str], works: Mapping[str, dict[str, str]]) -> tuple[list[str], bool]:
    """The retrieved works closest to the claim, chosen by code rather than by the examiner.

    Works screened out as being about something else are never "nearest": seven
    papers that merely share the word "graph" are not a literature this claim was
    checked against, and an examiner is not marked down for ignoring them.
    """
    from ..prior import claim_coverage, claim_support

    scored = []
    for cite, work in works.items():
        if work.get("relevance") == "off_topic":
            continue
        text = f"{work.get('title') or ''}. {work.get('abstract') or ''}"
        score = max((max(claim_support(claim, text)[0], claim_coverage(claim, text))
                     for claim in claims), default=0.0)
        scored.append((score, cite))
    scored.sort(key=lambda item: (-item[0], item[1]))
    near = [cite for _, cite in scored[:NEAREST_PRIOR]]
    reaches = any(work.get("relevance") == "on_topic" for work in works.values())
    return near, reaches


def _ask(client: Any, role: str, payload: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    from ..llm import complete_call
    from .ideation import parse_json_object

    completion = complete_call(client, json.dumps(payload, ensure_ascii=False, indent=1),
                               purpose=f"judgment_{role}", system=SYSTEM[role])
    return parse_json_object(getattr(completion, "text", "")), completion


def convene(state: Any, *, clients: Mapping[str, Any], workspace: Path | None) -> dict[str, Any]:
    """Run the four roles over the current subject and return proposed review rows.

    Raises ``LLMUnavailable`` when a role cannot be reached and ``ValueError``
    when a role's reply cannot be read; either way nothing is proposed.
    """
    subject = judgment_subject(state)
    works = _verified_works(state)
    claims = [row["claim"] for row in subject["findings"] if row.get("claim")]
    _screen(claims, works)
    near, reaches = _nearest_prior(claims, works)
    proposer = [
        {key: idea.get(key) for key in ("central_claim", "mechanism", "motivation",
                                        "why_it_may_matter", "novelty_hypothesis")}
        for row in getattr(state, "artifacts", []) or []
        if row.get("kind") == "discovery_idea"
        and str(row.get("id") or "") in {item["idea_id"] for item in subject["findings"]}
        for idea in [dict(row.get("idea") or {})]]
    brief = {
        "mission": subject["mission"],
        "findings_admitted_by_the_runtime": subject["findings"],
        "proposer_pitch_not_evidence": proposer,
        "nearest_prior_works": [works[cite] for cite in near],
        "other_verified_works": [{"cite_id": cite, "title": works[cite]["title"]}
                                 for cite in sorted(works) if cite not in near],
        "literature_pool_reaches_the_claim": reaches,
    }
    replies: dict[str, dict[str, Any]] = {}
    completions: dict[str, Any] = {}
    replies[EXAMINER], completions[EXAMINER] = _ask(clients[EXAMINER], EXAMINER, brief)
    table = [dict(row) for row in replies[EXAMINER].get("closest_prior") or []
             if isinstance(row, Mapping)]
    argued = {**brief, "prior_art_table": table,
              "what_the_examiner_found_new": replies[EXAMINER].get("what_is_new")}
    # Independent: neither sees the other's case.
    for role in (ADVOCATE, SKEPTIC):
        replies[role], completions[role] = _ask(clients[role], role, argued)
    # Order fixed by the subject, not by who wrote which case.
    pair = [replies[ADVOCATE], replies[SKEPTIC]]
    if int(subject["digest"][:2], 16) % 2:
        pair.reverse()
    cases = [{"case": str(item.get("case") or ""), "cites": list(item.get("refs") or [])}
             for item in pair]
    replies[ADJUDICATOR], completions[ADJUDICATOR] = _ask(clients[ADJUDICATOR], ADJUDICATOR, {
        **brief, "prior_art_table": table,
        "prior_art_examiner_grade": _grade(replies[EXAMINER].get("novelty")),
        "case_one": cases[0], "case_two": cases[1]})

    verdict = replies[ADJUDICATOR]
    grades = {
        "novelty": {EXAMINER: _grade(replies[EXAMINER].get("novelty")),
                    ADVOCATE: _grade(replies[ADVOCATE].get("novelty")),
                    SKEPTIC: _grade(replies[SKEPTIC].get("novelty")),
                    ADJUDICATOR: _grade(verdict.get("novelty"))},
        "importance": {ADVOCATE: _grade(replies[ADVOCATE].get("importance")),
                       SKEPTIC: _grade(replies[SKEPTIC].get("importance")),
                       ADJUDICATOR: _grade(verdict.get("importance"))},
    }
    if not grades["novelty"][ADJUDICATOR] or not grades["importance"][ADJUDICATOR]:
        raise ValueError("the adjudicator returned no usable grade")

    addressed = {str(row.get("cite_id") or "") for row in table}
    unaddressed = [cite for cite in near if cite not in addressed]
    caps = {"novelty": [], "importance": []}
    final = {
        "novelty": _lowest(grades["novelty"][EXAMINER], grades["novelty"][ADJUDICATOR]),
        "importance": _lowest(grades["importance"][ADVOCATE], grades["importance"][ADJUDICATOR]),
    }
    if unaddressed:
        caps["novelty"].append("examiner_did_not_address_the_nearest_prior_work")
    if not reaches:
        caps["novelty"].append("literature_pool_does_not_reach_the_claim")
    if caps["novelty"]:
        final["novelty"] = _cap(final["novelty"], "partial")

    models = {role: str(getattr(completions[role], "model", "") or "") for role in ROLES}
    independence = "distinct_models" if len(set(models.values())) > 1 else "same_model_roles"
    rank = _rank()
    # What grounds a verdict is what the verdict itself rests on: the examiner's
    # table for novelty, the adjudicator's own citations for either. An
    # advocate's reading list grounds nothing, an invented id is dropped, and so
    # is a real paper the survey screened out as being about something else.
    def verified(cites: Any) -> list[str]:
        return sorted({str(cite) for cite in cites or []
                       if str(cite) in works and works[str(cite)].get("relevance") != "off_topic"})

    grounded = {
        "novelty": verified(list(verdict.get("novelty_refs") or verdict.get("refs") or [])
                            + [row.get("cite_id") for row in table]),
        "importance": verified(verdict.get("importance_refs") or verdict.get("refs") or []),
    }
    improve = {"novelty": str(verdict.get("what_would_make_it_new") or ""),
               "importance": str(verdict.get("what_would_make_it_matter") or "")}

    transcript = ""
    if workspace is not None:
        folder = Path(workspace) / "reviews"
        folder.mkdir(parents=True, exist_ok=True)
        transcript = f"reviews/panel-{subject['digest'][:16]}.json"
        (Path(workspace) / transcript).write_text(json.dumps({
            "version": PANEL_VERSION, "epistemic": "GENERATED",
            "note": "a judgment, not a measurement; it adds no evidence",
            "subject": subject, "nearest_prior": near, "roles": replies, "models": models,
            "final": final, "caps": caps}, ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8")

    rows = []
    for name in ("importance", "novelty"):
        table_grades = grades[name]
        arguers = [table_grades[ADVOCATE], table_grades[SKEPTIC]]
        spread = (abs(rank[arguers[0]] - rank[arguers[1]]) if all(arguers) else 0)
        rows.append({
            "review_id": f"panel-{name}-{subject['digest'][:16]}",
            "dimension": name, "severity": "minor", "repair_type": "writing_only",
            "objection": "", "assessment": final[name], "source": SOURCE,
            "supporting_refs": grounded[name],
            "assessment_basis": str(verdict.get("decisive_argument") or "")[:600],
            "panel": {
                "version": PANEL_VERSION, "subject_digest": subject["digest"],
                "grades": table_grades, "spread": spread, "caps": caps[name],
                "unaddressed_prior": unaddressed if name == "novelty" else [],
                "unresolved_disagreement": str(verdict.get("unresolved_disagreement") or "")[:600],
                "improve": improve[name][:600],
                "prior_art_search_needed": [str(item) for item in
                                            replies[EXAMINER].get("prior_art_search_needed") or []][:6],
                "independence": independence, "models": models,
                "completion_digests": {role: str(getattr(completions[role], "digest", "") or "")
                                       for role in ROLES},
                "transcript": transcript,
            },
        })
    return {"rows": rows, "summary": {
        "version": PANEL_VERSION, "subject_digest": subject["digest"], "final": final,
        "independence": independence, "models": models, "transcript": transcript,
        "epistemic": "GENERATED"}}


def review(action: Any, state: Any, *, clients: Mapping[str, Any],
           workspace: Path | None) -> Any:
    """REVIEW with a model attached: the structural reviewer, plus the panel when asked.

    The structural objections are facts about the state and are recorded exactly
    as they are without a model. The panel only ever adds rows on top.
    """
    from dataclasses import replace

    from ..llm import LLMUnavailable
    from .executors import execute_review
    from .ideation import _refused, _service_unavailable
    from .kernel import readonly_view
    from .state import ScientificState

    view = ScientificState.from_dict(dict(readonly_view(state)))
    extra = dict(getattr(action, "extra", {}) or {})
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if extra.get("judgment_panel") and judgment_due(view):
        if any(clients.get(role) is None for role in ROLES):
            return _refused("research_model_not_connected")
        try:
            convened = convene(view, clients=clients, workspace=workspace)
        except LLMUnavailable as exc:
            return _service_unavailable(exc)
        except (ValueError, TypeError, KeyError) as exc:
            return _refused(f"panel_unparseable:{exc}", status="panel_unparseable")
        rows, summary = convened["rows"], convened["summary"]
    result = execute_review(SimpleNamespace(state=view), replace(
        action, extra={**extra, "reviews": list(extra.get("reviews") or []) + rows}))
    if summary:
        result.extras["judgment_panel"] = summary
        result.extras["not_evidence"] = "a panel's agreement never corroborates a claim"
    return result
