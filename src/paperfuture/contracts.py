"""JSON contracts shared by prompts, runtime validation and exported schemas.

These checks establish structural/provenance validity, not scientific truth.
"""
from __future__ import annotations

import math
from typing import Any


class ContractError(ValueError):
    pass


def string():
    return {"type": "string", "minLength": 1}


def enum(*values):
    return {"enum": list(values)}


def number(low, high, *, integer=False, nullable=False):
    result = {"type": "integer" if integer else "number", "minimum": low, "maximum": high}
    return {"anyOf": [result, {"type": "null"}]} if nullable else result


def array(item=None, minimum=0, maximum=10000):
    return {"type": "array", "items": item or string(), "minItems": minimum, "maxItems": maximum}


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def validate(value: Any, schema: dict, path="$") -> None:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return
            except ContractError:
                pass
        raise ContractError(f"{path}: no allowed type matched")
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ContractError(f"{path}: expected one of {schema['enum']}")
        return
    kind = schema["type"]
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "integer": type(value) is int,
               "number": type(value) in (float, int), "boolean": type(value) is bool,
               "null": value is None}
    if not matches[kind]:
        raise ContractError(f"{path}: expected {kind}")
    if kind == "object":
        fields = schema["properties"]
        if set(value) != set(fields):
            raise ContractError(f"{path}: missing={sorted(set(fields)-set(value))}, unexpected={sorted(set(value)-set(fields))}")
        for key, item in value.items():
            validate(item, fields[key], f"{path}.{key}")
    elif kind == "array":
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ContractError(f"{path}: invalid array length {len(value)}; expected {schema['minItems']}..{schema['maxItems']} items")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    elif kind == "string" and not value.strip():
        raise ContractError(f"{path}: empty string")
    elif kind in ("integer", "number"):
        if not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]:
            raise ContractError(f"{path}: outside [{schema['minimum']}, {schema['maximum']}]")


DIMENSIONS = ("problem", "assumption", "mechanism", "capability", "resource")
DELTA = obj(**{k: number(0, 10, integer=True, nullable=True) for k in DIMENSIONS})
DIMENSION_REASONS = obj(**{k: obj(reason=string(), evidence_ids=array(),
    source_evidence_ids=array(), target_evidence_ids=array(), comparison_evidence_ids=array()) for k in DIMENSIONS})
INTERVAL = obj(lower=number(0,100,integer=True), upper=number(0,100,integer=True))
EVIDENCE = array(string(), 1)
EXTRACTION = obj(relevance=number(0, 4, integer=True), quality=number(0, 4, integer=True),
    selection_reason=string(), problem=string(), mechanism=string(), mechanism_family=string(), assumptions=array(),
    capabilities=array(), limitations=array(), bottlenecks=array(), subtopics=array(string(), 1),
    evidence_ids=array(string(), 1, 12))
STAGE = obj(id=string(), label=string(), problem=string(), mechanism=string(), assumptions=array(),
    capabilities=array(), limitations=array(), bottlenecks=array(), paper_keys=array(string(), 1),
    evidence_ids=EVIDENCE)
EDGE = obj(id=string(), source=string(), target=string(),
    relation=enum("refine", "replace_mechanism", "relax_assumption", "change_objective", "transfer_setting", "combine", "limit_or_refute"),
    inherited=string(), changed=string(), bottleneck=string(), delta=DELTA,
    dimension_reasons=DIMENSION_REASONS, weight=number(0,100,integer=True,nullable=True),
    weight_interval=INTERVAL,
    rationale=string(), confidence=enum("high", "medium", "low", "unknown"),
    source_witness={**string(),"description":"An exact evidence quote ID from the source stage (E1, E2, etc. in the supplied input), never a stage ID such as s0."},
    target_witness={**string(),"description":"An exact evidence quote ID from the target stage (E1, E2, etc. in the supplied input), never a stage ID such as s1."},
    evidence_ids=array())
GRAPH = obj(stages=array(STAGE, 2), edges=array(EDGE), coverage_gaps=array())
STAGES = obj(stages=array(STAGE,1), coverage_gaps=array())
STAGE_MERGE = obj(groups=array(obj(member_stage_ids=array(string(),1),representative_stage_id=string(),
    comparisons=array(obj(stage_id=string(),same_problem={"type":"boolean"},same_mechanism={"type":"boolean"},
        same_assumptions={"type":"boolean"},same_capability_boundary={"type":"boolean"},reason=string(),evidence_ids=EVIDENCE))),2))
EDGES = obj(edges=array(EDGE), unsupported_pairs=array(obj(source=string(),target=string(),reason=string())))
EDGE_REVIEW = obj(reviews=array(obj(edge_id=string(),
    verdict=enum("supported", "unsupported", "uncertain"),
    relationship_reason=string(), evidence_ids=array(), inherited=string(), changed=string(), bottleneck=string(),
    rationale=string(), confidence=enum("high", "medium", "low", "unknown"),
    source_entailment=string(), target_entailment=string(),
    delta=DELTA, dimension_reasons=DIMENSION_REASONS,
    weight=number(0,100,integer=True,nullable=True), weight_interval=INTERVAL),1))
ASSUMPTION = obj(id=string(), statement=string(), origin=enum("literature", "hypothesis"), evidence_ids=array())
IDEA = obj(title=string(), problem=string(), mechanism=string(), difference=string(), minimal_test=string())
PROPOSAL = obj(source_stages=array(string(), 1), source_edges=array(), bottleneck=string(),
    historical_pattern=string(), operation=string(), target_delta=DELTA,
    target_weight=number(0,100,integer=True), target_interval=INTERVAL,
    magnitude_reason=string(), evidence_ids=EVIDENCE,
    causal_change=obj(existing_mechanism=string(), changed_principle=string(),
                     causal_chain=array(string(),2), why_not_module_stacking=string(), falsifier=string()),
    nearest_prior_art=array(obj(paper_key=string(), shared_elements=string(), mechanism_difference=string()),1),
    minimal_experiment=obj(baseline_paper_keys=array(string(),1), intervention=string(), control=string(),
                           observables=array(string(),1), falsification_rule=string(), confound=string()),
    learning_trace=array(obj(experience_id=string(), usage=enum("retain", "test", "repair", "avoid", "not_applicable"),
                            implication=string(), change_in_this_proposal=string())),
    assumption_decisions=array(obj(assumption_id=string(), disposition=enum("retain", "replace", "not_applicable"),
                                  replacement_ids=array(), reason=string())),
    memory_transfers=array(obj(experience_id=string(), applicability=string(), condition_differences=array(),
                              test=string(), evidence_ids=array())),
    retrieval_use=array(obj(paper_key=string(), disposition=enum("used", "not_relevant", "insufficient_evidence"),
                            reason=string(), evidence_ids=array())),
    assumptions=array(ASSUMPTION), idea=IDEA)
SCENARIO = obj(kind=enum("success", "partial", "failure"), conditions=array(string(), 1),
    mechanism_chain=array(string(), 1), observable_predictions=array(string(), 1),
    competing_explanation=string(), next_bottleneck=string(), evidence_ids=array(), assumption_ids=array())
SIMULATION = obj(epistemic_status=enum("conditional_simulation_not_evidence"), scenarios=array(SCENARIO, 3, 3))
OBJECTION = obj(id=string(), claim=string(), kind=enum("evidenced_conflict", "unknown", "taste"),
    severity=number(0, 1), condition=string(), evidence_ids=array(), assumption_ids=array(), repair=string())
CRITIQUE = obj(preserved_novelty=string(), objections=array(OBJECTION), missed_opportunities=array())
DISCUSSION = obj(epistemic_status=enum("conditional_simulation_not_evidence"),
    agreements=array(string()), disagreements=array(obj(claim=string(), with_model_role=string(), reason=string())),
    revised_predictions=array(obj(condition=string(), prediction=string(), evidence_ids=array(), assumption_ids=array())),
    discriminating_tests=array(string(),1), next_leap_implications=array(obj(implication=string(), assumption_ids=array(), evidence_ids=array()),1))
VALUE_SCORE = obj(score=number(0, 4, integer=True, nullable=True), reason=string(), evidence_ids=array())
RESOLUTION = obj(objection_id=string(), disposition=enum("resolved", "unresolved", "not_applicable"),
    conditions_match={"type": "boolean"}, reason=string(), evidence_ids=array())
ASSESSMENT = obj(novelty=VALUE_SCORE, importance=VALUE_SCORE, potential_gain=VALUE_SCORE,
    feasibility=enum("testable", "missing_conditions", "substantive_obstacle"),
    feasibility_reason=string(), mechanism_clear={"type": "boolean"}, validation_path_clear={"type": "boolean"},
    resolutions=array(RESOLUTION), remaining_unknowns=array(), next_question=string(),
    suggested_action=enum("revise", "continue_leap", "switch"), suggested_change=string(),
    parent_round=number(1, 4, integer=True), preserve_assumption_ids=array(),
    mechanism_signature=string(), assumption_signature=string(),
    mechanism_audit=obj(kind=enum("mechanistic_change","justified_interaction","already_known","integration_only","unclear"),
                       reason=string(), nearest_prior_art_keys=array(string(),1), evidence_ids=EVIDENCE,
                       discriminating_test={"type":"boolean"}),
    progress=obj(compared_round=number(0,4,integer=True), mechanism_changed={"type":"boolean"},
                 resolved_previous_issue={"type":"boolean"}, reason=string()),
    navigation=obj(target_stage_ids=array(string(),1), experience_ids=array(), rationale=string(),
        unresolved_issue=string(), change_axis=enum("bottleneck", "principle", "intervention", "test"),
        pass_condition=string(), queries=obj(bottleneck=string(), alternative=string(), falsifier=string())),
    model_disagreements=array(obj(issue=string(), positions=array(string(),2), resolution=string(),
                                 remains_uncertain={"type":"boolean"})))
RERANK = obj(results=array(obj(paper_key=string(), relevance=number(0, 4, integer=True),
    historical_relevance=number(0,4,integer=True),
    historical_role=enum("direct", "historical_enabler", "turning_point", "counterexample", "evaluation_foundation", "unrelated"),
    facets=array(), reason=string()), 1))
PROPOSAL_CHECK = obj(kind=enum("new_mechanism", "substantive_revision", "duplicate", "unclear"),
    compared_rounds=array(number(1,4,integer=True),1,4),
    changed_axes=array(enum("bottleneck", "principle", "intervention", "test")),
    reason=string(), unresolved_issue_addressed=string())
FINAL_REVIEW = obj(candidate_round=number(1,4,integer=True),
    resolutions=array(obj(objection_id=string(), disposition=enum("resolved", "unresolved", "not_applicable"),
        conditions_match={"type":"boolean"}, reason=string(), evidence_ids=array())),
    assumption_findings=array(obj(assumption_id=string(),
        status=enum("retained", "replaced_in_later_candidate", "uncertain", "refuted_by_evidence"),
        reason=string(), evidence_ids=array())),
    feasibility=enum("testable", "missing_conditions", "substantive_obstacle"), feasibility_reason=string(),
    remaining_unknowns=array(), prior_art_status=enum("unchanged", "already_known", "uncertain"),
    prior_art_reason=string(), prior_art_evidence_ids=array(), selection_reason=string())
SCHEMAS = {"extract": EXTRACTION, "rerank": RERANK, "graph": GRAPH,
           "graph_stages": STAGES, "graph_merge":STAGE_MERGE,"graph_edges": EDGES, "graph_review": EDGE_REVIEW, "propose": PROPOSAL,
           "simulate": SIMULATION, "critique": CRITIQUE, "discuss": DISCUSSION, "assess": ASSESSMENT,
           "proposal_check":PROPOSAL_CHECK, "final_review":FINAL_REVIEW}
SCHEMAS['research_tools']=obj(requests=array(obj(
    tool=enum('search_papers','read_passages','compare_stages'),query=string(),domains=array(),
    paper_keys=array(),stage_ids=array(),purpose=enum('closest_prior_art','falsifier','mechanism','transfer_conditions'),
    reason=string()),0,6),stop_reason=string())


def require_refs(ids: list, available: dict | set, context: str, nonempty=False) -> None:
    if nonempty and not ids:
        raise ContractError(f"{context}: evidence is required")
    if len(ids) != len(set(ids)):
        raise ContractError(f"{context}: duplicate identifiers")
    missing = set(ids) - set(available)
    if missing:
        raise ContractError(f"{context}: unknown references {sorted(missing)}; supplied identifiers include {sorted(available)[:60]}")
