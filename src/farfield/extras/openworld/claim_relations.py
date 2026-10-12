"""Limited, checkable relations between two registered propositions.

The project needed one thing it did not have: a way to say that a measured
proposition *answers* the root question when the two are not the same sentence.
Without it a confirmed measurement of "removing edge (0,1) lowers mean_degree"
left the root question "does removing edge (0,1) raise mean_degree?" recorded as
unresolved — the evidence existed, the answer was negative and adequate, and no
projection could say so.

The dangerous way to fix that is a similarity score over prose. This module
does the opposite. It trusts exactly one mapping from text to structure: the
kernel's own controlled sentence for a registered operation, recognised by
round-tripping the parse back through ``canonical_claim``. A sentence that does
not re-render to itself has no structure here and every relation over it is
``unknown``. A model may propose a relation; only :func:`validate_claim_relation`
can mark one ``validated``, and it recomputes rather than reads the proposal.

The relation vocabulary is deliberately three words wide:

``equivalent``
    Same operation, parameters, observable, conditions, quantifier and
    direction. The two sentences are the same proposition.
``incompatible``
    Everything matches except the direction. Because both decision margins are
    positive by registration, a statistic that clears one direction's margin
    cannot clear the other's: supporting either proposition refutes the other.
``unknown``
    Everything else, including every relation between prose this module cannot
    structure. Unknown is a real answer and is never upgraded by repetition.

What this is not: a natural-language entailment engine, a causal identification
rule, or a way to carry a result from one scope to another. A mean over five
settings does not establish each setting; a relation between two propositions
says nothing about any condition neither of them names.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping

from .experiment import OPERATIONS, canonical_claim

RULE_VERSION = "claim-relations-v1"

EQUIVALENT = "equivalent"
INCOMPATIBLE = "incompatible"
UNKNOWN = "unknown"

VALIDATED = "validated"
REJECTED = "rejected"

#: The only scope a structured claim can carry. The sentences this module
#: parses are statements about one bound world under one registered operation.
SCOPE = "registered_operation_on_attested_world"
QUANTIFIER = "the_single_bound_world"

_GRAPH = re.compile(
    r"^Removing edges (?P<remove>\[.*?\]) and adding edges (?P<add>\[.*?\]) "
    r"(?P<direction>raises|lowers) (?P<metric>[A-Za-z_][A-Za-z0-9_]*) on the bound graph\.$")
_REPO = re.compile(
    r"^Replacing (?P<old>'[^']*'|\"[^\"]*\") with (?P<new>'[^']*'|\"[^\"]*\") "
    r"(?P<direction>raises|lowers) tests_passed on the bound repository\.$")
_TABLE = re.compile(
    r"^Nearest-centroid classification using (?P<feature>.+?) predicts (?P<label>.+?) "
    r"(?P<direction>more|less) accurately than the majority baseline on heldout rows "
    r"\(seed=(?P<seed>-?\d+), test_fraction=(?P<fraction>[0-9.eE+-]+)\)\.$")

_DIRECTION = {"raises": "treatment_higher", "lowers": "treatment_lower",
              "more": "treatment_higher", "less": "treatment_lower"}


def _edges(blob: str) -> list[list[int]] | None:
    try:
        value = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, list):
        return None
    for edge in value:
        if (not isinstance(edge, list) or len(edge) != 2
                or any(type(node) is not int for node in edge)):
            return None
    return value


def _protocol_from_text(text: str) -> dict[str, Any] | None:
    """Recover the protocol the kernel would have rendered as ``text``."""
    match = _GRAPH.match(text)
    if match:
        add, remove = _edges(match["add"]), _edges(match["remove"])
        if add is None or remove is None:
            return None
        return {"operation": "graph_edit",
                "parameters": {"add_edges": add, "remove_edges": remove},
                "metric": match["metric"],
                "expected_direction": _DIRECTION[match["direction"]]}
    match = _REPO.match(text)
    if match:
        try:
            old, new = ast.literal_eval(match["old"]), ast.literal_eval(match["new"])
        except (ValueError, SyntaxError):
            return None
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return {"operation": "repository_replace",
                "parameters": {"old": old, "new": new},
                "metric": "tests_passed",
                "expected_direction": _DIRECTION[match["direction"]]}
    match = _TABLE.match(text)
    if match:
        try:
            fraction = float(match["fraction"])
        except ValueError:
            return None
        return {"operation": "table_classification",
                "parameters": {"feature": match["feature"], "label": match["label"],
                               "test_fraction": fraction, "seed": int(match["seed"])},
                "metric": "accuracy",
                "expected_direction": _DIRECTION[match["direction"]]}
    return None


def structured_claim(text: Any) -> dict[str, Any] | None:
    """Structure ``text`` only if it *is* a kernel-rendered operational claim.

    The round-trip is the whole guarantee. Parsing alone would accept a prose
    sentence that merely resembles the template; re-rendering the parse through
    :func:`canonical_claim` and demanding byte equality accepts only sentences
    the kernel itself would have written for that registration.
    """
    sentence = str(text or "").strip()
    if not sentence:
        return None
    protocol = _protocol_from_text(sentence)
    if protocol is None or protocol["operation"] not in OPERATIONS:
        return None
    try:
        if canonical_claim(protocol) != sentence:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    klass, metrics = OPERATIONS[protocol["operation"]]
    if protocol["metric"] not in metrics:
        return None
    params = dict(protocol["parameters"])
    conditions = {k: params.pop(k) for k in ("seed", "test_fraction") if k in params}
    return {
        "rule_version": RULE_VERSION,
        "source": "kernel_controlled_sentence",
        "operation": protocol["operation"],
        "world_class": klass,
        "treatment": params,
        "control": "the registered operator control on the same frozen world",
        "observable": protocol["metric"],
        "unit": protocol["metric"],
        "direction": protocol["expected_direction"],
        "conditions": conditions,
        "quantifier": QUANTIFIER,
        "scope": SCOPE,
        "statement": sentence,
    }


def structured_from_execution(execution: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Structure a frozen protocol by the sentence the kernel renders for it."""
    protocol = dict(execution or {})
    if protocol.get("operation") not in OPERATIONS:
        return None
    try:
        return structured_claim(canonical_claim(protocol))
    except (KeyError, TypeError, ValueError):
        return None


_IDENTITY = ("operation", "world_class", "treatment", "control", "observable",
             "unit", "conditions", "quantifier", "scope")


def relate(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare two structured claims under the single rule this module has."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {"relation": UNKNOWN, "rule_version": RULE_VERSION,
                "reason": "at_least_one_side_has_no_trusted_structure", "premises": []}
    differing = [name for name in _IDENTITY if left.get(name) != right.get(name)]
    if differing:
        return {"relation": UNKNOWN, "rule_version": RULE_VERSION,
                "reason": "claims_differ_on:" + ",".join(sorted(differing)),
                "premises": []}
    premises = [
        f"operation={left['operation']}",
        f"observable={left['observable']}",
        f"quantifier={left['quantifier']}",
        f"scope={left['scope']}",
    ]
    if left.get("direction") == right.get("direction"):
        return {"relation": EQUIVALENT, "rule_version": RULE_VERSION,
                "reason": "identical_registered_comparison_and_direction",
                "premises": premises}
    return {
        "relation": INCOMPATIBLE,
        "rule_version": RULE_VERSION,
        "reason": "same_comparison_opposite_direction_with_positive_registered_margins",
        "premises": premises + [
            "every registered decision margin is positive, so one direction's "
            "corroborating statistic falls outside the other's",
        ],
    }


def relation_between(left_text: Any, right_text: Any) -> dict[str, Any]:
    """Relation between two claim *sentences*, structured by the rule above."""
    return relate(structured_claim(left_text), structured_claim(right_text))


def invert_outcome(outcome: Any, relation: str) -> str:
    """Read an outcome recorded against one claim as an outcome for the other.

    Only an ``incompatible`` pair flips: the decision rule is symmetric about
    zero and both margins are positive, so clearing one claim's margin fails the
    other's. Nothing is inverted for ``unknown`` — there is no mapping to apply.
    """
    value = str(outcome or "")
    if relation == EQUIVALENT:
        return value
    if relation == INCOMPATIBLE:
        return {"supports": "weakens", "weakens": "supports"}.get(value, "")
    return ""


def validate_claim_relation(proposal: Mapping[str, Any] | None,
                            state: Any = None) -> dict[str, Any]:
    """Grade a *proposed* relation by recomputing it.

    A worker may propose ``relation``; it never survives into the result. The
    status is ``validated`` only when this module derives the same relation from
    both sentences' own structure, ``rejected`` when it derives a different one,
    and ``unknown`` when it cannot structure either side. ``state`` is accepted
    so callers can pass the project without unpacking it; premises name the
    claims, never a model's confidence in them.
    """
    row = dict(proposal or {})
    left_text, right_text = row.get("left_claim"), row.get("right_claim")
    computed = relation_between(left_text, right_text)
    claimed = str(row.get("relation") or "")
    result = {
        "status": UNKNOWN,
        "relation": computed["relation"],
        "premise_ids": [str(item) for item in row.get("premise_ids") or ()],
        "premises": list(computed["premises"]),
        "scope": SCOPE,
        "rule_version": RULE_VERSION,
        "reason": computed["reason"],
        "proposed_relation": claimed,
        "epistemic": "DERIVED",
    }
    if computed["relation"] == UNKNOWN:
        return result
    if claimed and claimed != computed["relation"]:
        result["status"] = REJECTED
        result["reason"] = f"proposed_{claimed}_but_rule_derives_{computed['relation']}"
        return result
    result["status"] = VALIDATED
    return result
