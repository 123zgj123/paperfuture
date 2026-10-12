"""Claim-derived experiment contracts and world-match classification.

Not a controller. OpenWorld still owns the next action. This module states
what evidence would change an idea's status, and whether a world can execute
that contract. The contract is GENERATED and derived: replay rebuilds it from
the idea and its feedback. Nothing here writes WORLD or belief.

See reports/ARCHITECTURE_DECISION.md for why CompleteIdea, WorldRequirement, ClaimSpec
and diagnosis were not extended to hold this invariant.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from ..world import (
    WorldFixture,
    WorldRequirement,
    digest_files,
    infer_requirement,
    load_fixture,
    match_world,
    requirement_compatible,
)

# Study stages on PROBE / VERIFY. Not new action types.
# `pilot` is the legacy alias of a scientific cheap run.
PILOT = "pilot"
CALIBRATION = "calibration"
SCIENTIFIC_PILOT = "scientific_pilot"
CONFIRMATORY = "confirmatory"
REPLICATION = "replication"
STUDY_STAGES = (PILOT, CALIBRATION, SCIENTIFIC_PILOT, CONFIRMATORY, REPLICATION)
CHEAP_STAGES = frozenset({PILOT, CALIBRATION, SCIENTIFIC_PILOT})
SCIENTIFIC_CHEAP_STAGES = frozenset({PILOT, SCIENTIFIC_PILOT})

PILOT_POSITIVE = "PILOT_POSITIVE"
PILOT_NEGATIVE = "PILOT_NEGATIVE"
PILOT_INCONCLUSIVE = "PILOT_INCONCLUSIVE"
PILOT_VERDICTS = (PILOT_POSITIVE, PILOT_NEGATIVE, PILOT_INCONCLUSIVE)

MATCHED_EXISTING_WORLD = "MATCHED_EXISTING_WORLD"
MATCH_REQUIRES_INSTRUMENTATION = "MATCH_REQUIRES_INSTRUMENTATION"
CAN_ACQUIRE_WORLD = "CAN_ACQUIRE_WORLD"
CAN_BUILD_WORLD = "CAN_BUILD_WORLD"
RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
UNSUPPORTED_WORLD = "UNSUPPORTED_WORLD"
CAPABILITY_GAP = "CAPABILITY_GAP"
MATCH_STATUSES = (
    MATCHED_EXISTING_WORLD,
    MATCH_REQUIRES_INSTRUMENTATION,
    CAN_ACQUIRE_WORLD,
    CAN_BUILD_WORLD,
    RESOURCE_BLOCKED,
    UNSUPPORTED_WORLD,
    CAPABILITY_GAP,
)

REPOSITORY = "repository"
DATASET = "dataset"
SIMULATION = "simulation"
INFERENCE = "inference"
WORLD_CLASSES = (REPOSITORY, DATASET, SIMULATION, INFERENCE)

# V3 is a validation class. Pilot is an execution stage. A pilot may use V3.
V3_CHEAP_EXTERNAL = "V3_CHEAP_EXTERNAL"
V4_CONFIRMATORY = "V4_CONFIRMATORY"
EXPERIMENT = "EXPERIMENT"
WORLD_EXTENSION = "WORLD_EXTENSION"
CONTRACT_KINDS = (EXPERIMENT, WORLD_EXTENSION)

CONSTRUCT_TEMPLATE = "construct_template"
TEMPLATE_SOURCE = {
    REPOSITORY: "farfield-template:repository-v1",
    DATASET: "farfield-template:dataset-v1",
    SIMULATION: "farfield-template:simulation-v1",
    INFERENCE: "farfield-template:inference-v1",
}
CLASS_SCHEMA = {
    REPOSITORY: "repository",
    DATASET: "numeric_table",
    SIMULATION: "undirected_graph",
    INFERENCE: "inference_job",
}

_WETLAB = re.compile(
    r"\b(wet[- ]?lab|petri|pipette|in vivo|in vitro|animal trial|physical robot|"
    r"human subject|mri scanner|telescope)\b",
    re.I,
)
_HEAVY = re.compile(
    r"\b(70b|70 b|hundred[- ]billion|multi[- ]gpu|cluster[- ]years?|private dataset|"
    r"unavailable data)\b",
    re.I,
)
_REPO = re.compile(
    r"\b(repository|codebase|test suite|unit tests?|checkout|git repo|coding agent|"
    r"software engineering task)\b",
    re.I,
)
_INFER = re.compile(
    r"\b(inference[- ]time|decoding strategy|sampling temperature|prompt template|"
    r"tiny training|logit)\b",
    re.I,
)
_SIM = re.compile(
    r"\b(simulation|synthetic instance|random seed|graph algorithm|optimisation|"
    r"optimization|compression ratio|numerical)\b",
    re.I,
)
_DATA = re.compile(
    r"\b(dataset|benchmark|train[- ]test split|held[- ]out split|classification|"
    r"numeric table|baseline metric)\b",
    re.I,
)
_TEMPLATE_TOO_SMALL = re.compile(
    r"\b(fifty|hundred|long[- ]horizon|long tasks|coding agent|swe[- ]bench|"
    r"live[- ]swe|cluster)\b",
    re.I,
)

FAILURE_TO_OPERATOR = {
    RESOURCE_BLOCKED: "SCOPE_SHIFT",
    UNSUPPORTED_WORLD: "SCOPE_SHIFT",
    MATCH_REQUIRES_INSTRUMENTATION: "MEASUREMENT_REPAIR",
    CAPABILITY_GAP: "MEASUREMENT_REPAIR",
    "PILOT_NEGATIVE": "",
    "PILOT_INCONCLUSIVE": "DISCRIMINATIVE_TEST",
    "no_acquisition_recipe": "REFORMULATE",
}

EXPLORATION_ARMS = (
    "farfield_remote_bootstrap",
    "direct_llm_brainstorm",
    "local_literature_expansion",
    "nearest_concept",
    "random_concept",
)


@dataclass(frozen=True)
class ExperimentRequirement:
    """What evidence would change this idea's status.

    Compiled from claim semantics. A world may satisfy it; a world may not
    write it. Epistemic status is always GENERATED.
    """

    claim_id: str
    claim_type: str
    hypothesis: str
    falsifier: str
    question_id: str = ""
    intervention: str = ""
    control_condition: str = ""
    observables: tuple[str, ...] = ()
    measurement: str = ""
    primary_metric: str = ""
    required_entities: tuple[str, ...] = ()
    required_operations: tuple[str, ...] = ()
    reset_requirements: str = ""
    isolation_requirements: str = ""
    randomization_requirements: str = ""
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    replication_unit: str = ""
    minimum_sample_or_task_count: int = 0
    matched_compute_requirement: str = ""
    major_confounders: tuple[str, ...] = ()
    required_external_assets: tuple[str, ...] = ()
    success_criterion: str = ""
    failure_criterion: str = ""
    world_class: str = ""
    required_world_capabilities: tuple[str, ...] = ()
    schema: str = ""
    object_type: str = ""
    epistemic: str = "GENERATED"
    execution: dict[str, Any] = field(default_factory=dict)
    kind: str = EXPERIMENT
    extension: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                out[item.name] = list(value)
            elif isinstance(value, dict):
                out[item.name] = dict(value)
            else:
                out[item.name] = value
        return out

    @classmethod
    def from_dict(cls, row: Mapping[str, Any] | None) -> "ExperimentRequirement":
        raw = dict(row or {})
        def texts(name: str) -> tuple[str, ...]:
            value = raw.get(name) or ()
            if isinstance(value, str):
                value = [value]
            return tuple(str(item) for item in value if str(item).strip())
        resources = raw.get("resource_requirements")
        return cls(
            claim_id=str(raw.get("claim_id") or ""),
            claim_type=str(raw.get("claim_type") or ""),
            hypothesis=str(raw.get("hypothesis") or ""),
            falsifier=str(raw.get("falsifier") or ""),
            question_id=str(raw.get("question_id") or ""),
            intervention=str(raw.get("intervention") or ""),
            control_condition=str(raw.get("control_condition") or ""),
            observables=texts("observables"),
            measurement=str(raw.get("measurement") or ""),
            primary_metric=str(raw.get("primary_metric") or ""),
            required_entities=texts("required_entities"),
            required_operations=texts("required_operations"),
            reset_requirements=str(raw.get("reset_requirements") or ""),
            isolation_requirements=str(raw.get("isolation_requirements") or ""),
            randomization_requirements=str(raw.get("randomization_requirements") or ""),
            resource_requirements=dict(resources) if isinstance(resources, Mapping) else {},
            replication_unit=str(raw.get("replication_unit") or ""),
            minimum_sample_or_task_count=int(raw.get("minimum_sample_or_task_count") or 0),
            matched_compute_requirement=str(raw.get("matched_compute_requirement") or ""),
            major_confounders=texts("major_confounders"),
            required_external_assets=texts("required_external_assets"),
            success_criterion=str(raw.get("success_criterion") or ""),
            failure_criterion=str(raw.get("failure_criterion") or ""),
            world_class=str(raw.get("world_class") or ""),
            required_world_capabilities=texts("required_world_capabilities"),
            schema=str(raw.get("schema") or ""),
            object_type=str(raw.get("object_type") or ""),
            epistemic="GENERATED",
            execution=dict(raw.get('execution') or {}),
            kind=str(raw.get("kind") or EXPERIMENT) if str(raw.get("kind") or EXPERIMENT) in CONTRACT_KINDS else EXPERIMENT,
            extension=dict(raw.get("extension") or {}),
        )

    def world_requirement(self) -> WorldRequirement | None:
        """Object-identity half, for the existing matcher. Not the whole contract."""
        if not (self.object_type or self.schema):
            return None
        return WorldRequirement(
            object_type=self.object_type,
            schema=self.schema,
            operations=self.required_operations,
            metrics=(self.primary_metric,) if self.primary_metric else (),
            environment=(self.world_class,) if self.world_class else (),
        )


def _blob(idea: Any, extra: str = "") -> str:
    parts = [
        getattr(idea, "research_problem", ""),
        getattr(idea, "central_claim", ""),
        getattr(idea, "falsifier", ""),
        getattr(idea, "validation_direction", ""),
        getattr(idea, "optional_experiment_direction", ""),
        extra,
    ]
    return " ".join(str(item or "") for item in parts)


def classify_world_class(idea: Any, *, root_question: str = "") -> str:
    """World class from claim semantics. The catalog does not vote."""
    text = _blob(idea, root_question)
    if _WETLAB.search(text):
        return ""
    if _REPO.search(text):
        return REPOSITORY
    if _INFER.search(text):
        return INFERENCE
    if _SIM.search(text):
        return SIMULATION
    if _DATA.search(text):
        return DATASET
    inferred = infer_requirement(text)
    if inferred is None:
        return ""
    from ..schemas import SCHEMAS

    spec = SCHEMAS.get(inferred.schema)
    return str(getattr(spec, "world_class", "") or "")


def compile_experiment_requirement(
    idea: Any,
    *,
    idea_id: str = "",
    question_id: str = "",
    feedback: Mapping[str, Any] | None = None,
    root_question: str = "",
) -> ExperimentRequirement:
    """Deterministic compiler. No model. Always GENERATED."""
    fb = dict(feedback or {})
    text = _blob(idea, root_question)
    world_class = classify_world_class(idea, root_question=root_question)
    from .experiment import OPERATIONS
    execution = dict(getattr(idea, 'experiment', {}) or {})
    # A study's operational sentence carries the digest of its registered setting
    # list, so that two studies over different settings can never render the same
    # proposition. No proposer can compute that digest, so the kernel writes the
    # sentence rather than asking for it back. Deriving it is also the stricter
    # choice: there is no sentence for a proposal to smuggle past the check.
    from .study import study_of
    if study_of(execution) is not None:
        from .experiment import canonical_claim
        try:
            execution['hypothesis'] = canonical_claim(execution)
        except (KeyError, TypeError, ValueError):
            pass
    kind, extension = _contract_parts(execution)
    if isinstance(execution.get('operation'), str) and execution['operation'] in OPERATIONS and not _WETLAB.search(text):
        world_class = OPERATIONS[execution['operation']][0]
    inferred = infer_requirement(text)
    schema = CLASS_SCHEMA.get(world_class, "")
    object_type = world_class
    if inferred is not None:
        schema = schema or inferred.schema
        object_type = inferred.object_type or object_type
    if kind == WORLD_EXTENSION:
        if not extension.get("action") and isinstance(execution.get("operation"), str):
            extension = {**extension, "action": execution["operation"]}
        if world_class not in WORLD_CLASSES:
            world_class = SIMULATION
            schema = schema or CLASS_SCHEMA[SIMULATION]
            object_type = object_type or SIMULATION
    resources: dict[str, Any] = {"gpu": False, "network": False, "max_runtime_s": 30}
    if _HEAVY.search(text):
        resources.update(gpu=True, infeasible=True, reason="declared_compute_or_data_unavailable")
    operations = {
        REPOSITORY: ("checkout", "modify", "run_tests"),
        DATASET: ("split", "baseline", "metric"),
        SIMULATION: ("instance", "seed", "intervene"),
        INFERENCE: ("decode", "seed", "metric"),
    }.get(world_class, ())
    capabilities = {
        REPOSITORY: ("repository", "checkout", "files", "tests", "commands", "reset"),
        DATASET: ("table", "rows", "numeric", "split", "metric", "reset"),
        SIMULATION: ("graph", "nodes", "edges", "seed", "repeat", "reset"),
        INFERENCE: ("inference", "decode", "prompt", "seed", "metrics", "reset"),
    }.get(world_class, ())
    metric = {
        REPOSITORY: "tests_passed",
        DATASET: "accuracy",
        SIMULATION: "objective",
        INFERENCE: "score",
    }.get(world_class, "primary_metric")
    metric = str(execution.get('metric') or metric)
    sample = execution.get('min_sample_size', 1)
    sample = sample if type(sample) is int and sample > 0 else 1
    return ExperimentRequirement(
        claim_id=str(idea_id or ""),
        question_id=question_id,
        claim_type=str(getattr(idea, "claim_type", "") or ""),
        hypothesis=str(getattr(idea, "central_claim", "") or ""),
        falsifier=str(getattr(idea, "falsifier", "") or ""),
        intervention=str(getattr(idea, "mechanism", "") or ""),
        control_condition=str(
            fb.get("strongest_competing_explanation")
            or getattr(idea, "competing_explanation", "")
            or ""
        ),
        observables=(metric,),
        measurement=str(getattr(idea, "validation_direction", "") or metric),
        primary_metric=metric,
        required_entities=(object_type,) if object_type else (),
        required_operations=operations,
        reset_requirements="re-read attested bytes; do not keep writes across runs",
        isolation_requirements="each run uses its own working copy under the workspace",
        randomization_requirements="declare seed; default 0",
        resource_requirements=resources,
        replication_unit="seed" if world_class in {SIMULATION, INFERENCE} else "split",
        minimum_sample_or_task_count=sample,
        matched_compute_requirement="same seed family, same template runner",
        major_confounders=("unequal compute", "result-driven metric change"),
        required_external_assets=(),
        success_criterion=str(getattr(idea, "validation_direction", "") or "")
        or f"treatment {metric} exceeds control",
        failure_criterion=str(getattr(idea, "falsifier", "") or "")
        or f"treatment {metric} does not exceed control",
        world_class=world_class,
        required_world_capabilities=capabilities,
        schema=schema,
        object_type=object_type,
        execution=execution,
        kind=kind,
        extension=extension,
    )


def _contract_parts(execution: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """One contract language. Extension is a proposed World mutation, not a second protocol."""
    blob = dict(execution or {})
    adds = dict(blob.get("adds") or blob.get("extension") or {})
    implementation = blob.get("implementation")
    if str(blob.get("kind") or "") == WORLD_EXTENSION or adds or (
            isinstance(implementation, str) and implementation.strip()):
        if isinstance(implementation, str) and implementation.strip():
            adds = {**adds, "implementation": implementation}
        return WORLD_EXTENSION, adds
    return EXPERIMENT, {}


def _units_per_arm(fixture: Any, operation: str) -> dict[str, Any]:
    """How many independent units one arm of this operator yields on this world.

    Different parameters, seeds or file names never raise this number; only more
    independent objects do. It is the ceiling on a registered ``min_sample_size``.
    """
    if operation == "graph_edit":
        return {"count": 1, "basis": "one independent graph fixture per arm; "
                                     "nodes and edges are not replicates"}
    if operation == "repository_replace":
        return {"count": 1, "basis": "one independent repository checkout per arm; "
                                     "assertions are not replicates"}
    if operation == "table_classification":
        try:
            rows = len(json.loads((Path(fixture.root) / "table.json").read_text())["rows"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {"count": None, "basis": "int(rows * test_fraction) heldout rows per arm"}
        return {"count": None, "rows": rows,
                "basis": f"int({rows} * test_fraction) heldout rows per arm, "
                         "paired across both arms"}
    return {"count": None, "basis": "declared by the admitted operator receipt"}


def instrument_gap(state: Any, req: Any) -> dict[str, Any]:
    """What a proposal asked to measure, what can be measured, and the difference.

    A live mission asked whether a chord improves a graph's *robustness*. The
    project could not operationalise it, reported `missing_executable_protocol`,
    spent its discovery budget and stopped — and nowhere in the state did it say
    the actual problem: the bound instruments read ``mean_degree`` and
    ``edge_count``, and nothing measures component survival under edge deletion.
    "Not operationalised" is true of every idea before it is compiled; it is not
    a diagnosis.

    Two sides, kept apart on purpose. What was *requested* is the proposal's own
    wording and stays GENERATED — this function never decides what the prose
    meant. What is *available* is read from attested world interfaces where a
    world is bound, and otherwise from the kernel's own operator catalogue for
    the declared world class. The difference between a declaration and an
    attestation is checkable; nothing here is inferred.
    """
    from .experiment import OPERATIONS
    from ..world import load_fixture

    requested = [str(name) for name in (req.observables or ()) if str(name).strip()]
    primary = str(req.primary_metric or "")

    operations: dict[str, list[str]] = {}
    worlds: list[str] = []
    source = "registered_operator_catalogue_for_the_declared_world_class"
    for row in getattr(state, "world_versions", []) or ():
        if row.get("deprecated") or not row.get("attested") or not row.get("path"):
            continue
        try:
            view = world_capability_view(load_fixture(Path(row["path"])))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not view.get("interfaces_attested"):
            continue
        worlds.append(str(row.get("id") or ""))
        for name, spec in (view.get("callable_operations") or {}).items():
            operations.setdefault(name, []).extend(
                str(metric) for metric in spec.get("metrics") or ())
        source = "attested_world_interfaces"
    known_classes = sorted({world_class for world_class, _ in OPERATIONS.values()})
    class_supported = True
    if not operations:
        for name, (world_class, metrics) in OPERATIONS.items():
            if not req.world_class or world_class == req.world_class:
                operations[name] = list(metrics)
        # A declared world class nothing implements is a different shortfall
        # from an observable no instrument reads, and saying "the available
        # instruments read nothing" for it is false: they read plenty, just not
        # for this kind of object.
        class_supported = bool(operations)

    offered = sorted({metric for metrics in operations.values() for metric in metrics})
    missing = sorted({name for name in requested if name not in offered})
    unmapped = bool(primary) and primary not in offered
    if not (missing or unmapped):
        return {}
    return {
        "requested": {
            "observables": requested,
            "primary_metric": primary,
            "measurement": str(req.measurement or "")[:400],
            "epistemic": "GENERATED",
            "note": "the proposal's own words; this is not a reading of what they mean",
        },
        "available": {
            "operations": {name: sorted(set(metrics)) for name, metrics in sorted(operations.items())},
            "observables": offered,
            "world_ids": worlds,
            "source": source,
            "world_class": str(req.world_class or ""),
            "world_class_supported": class_supported,
            "known_world_classes": known_classes,
        },
        "missing_observables": missing,
        "primary_metric_unmapped": unmapped,
        "gap": ("no_instrument_is_registered_for_the_declared_world_class"
                if not class_supported else
                "requested_observable_is_not_offered_by_any_available_instrument"),
    }


def world_capability_view(fixture: Any) -> dict[str, Any]:
    """Public callable interfaces on attested bytes; never control-case answers."""
    import ast
    import copy
    from ..schemas import SCHEMAS
    from .experiment import OPERATIONS, admitted_receipt, digest, safe_identifier
    from .worldio import is_attested

    schema = str(getattr(fixture, "schema", "") or "")
    spec = SCHEMAS.get(schema)
    world_class = str(getattr(spec, "world_class", "") or "")
    caps = tuple(sorted(getattr(spec, "capabilities", ()) or ()))
    operations: dict[str, Any] = {}
    try:
        attested = is_attested(fixture)
    except (OSError, ValueError, AttributeError, TypeError):
        attested = False
    files = set(getattr(fixture, "files", ()) or ())
    if attested:
        # These are the public signatures of existing built-in implementations.
        # A schema label alone does not supply their required data files.
        builtin_inputs = {
            'graph_edit': ({'graph.json'}, {'add_edges': {'type': 'array'}, 'remove_edges': {'type': 'array'}}, []),
            'repository_replace': ({'repo.py', 'test_repo.py'}, {'old': {'type': 'string'}, 'new': {'type': 'string'}}, ['old', 'new']),
            'table_classification': ({'table.json'}, {'feature': {'type': 'string'}, 'label': {'type': 'string'},
                'test_fraction': {'type': 'number', 'exclusiveMinimum': 0, 'exclusiveMaximum': 1},
                'seed': {'type': 'integer'}}, ['feature', 'label', 'test_fraction', 'seed']),
        }
        for name, (needed, properties, required) in builtin_inputs.items():
            if needed <= files:
                operations[name] = {
                    'source': 'registered', 'operation': name, 'metrics': list(OPERATIONS[name][1]),
                    'parameters_schema': {'type': 'object', 'properties': properties,
                                          'required': required, 'additionalProperties': False},
                    'schema_basis': 'registered_operator_interface',
                    # What this operator can actually supply on *this* world, so
                    # a proposal does not register a sample floor nothing can
                    # meet. A live run guessed five and had every registration
                    # refused, with nothing telling it the number available.
                    'independent_units_per_arm': _units_per_arm(fixture, name),
                }
        receipt = admitted_receipt(fixture)
        action = str(receipt.get('action') or '')
        source_name = f'admitted/{action}.py'
        if (receipt.get('admitted') and safe_identifier(action)
                and {'admitted_receipt.json', source_name} <= files):
            try:
                source = (Path(fixture.root) / source_name).read_text(encoding='utf-8')
                if digest(source) != receipt.get('implementation_digest'):
                    raise ValueError('admitted implementation identity mismatch')
                declared = receipt.get('parameters_schema')
                if isinstance(declared, Mapping) and declared.get('type') == 'object':
                    schema_inputs = copy.deepcopy(dict(declared))
                    basis = 'admitted_receipt_public_parameters_schema'
                    complete = True
                else:
                    # Inspect only act's parameter accesses, never execute the
                    # implementation or parse independent_controls.json.
                    tree = ast.parse(source)
                    act = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name == 'act')
                    argument = act.args.args[0].arg
                    keys: set[str] = set()
                    required_keys: set[str] = set()
                    complete = True
                    for node in ast.walk(act):
                        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == argument:
                            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                                keys.add(node.slice.value)
                                required_keys.add(node.slice.value)
                            else:
                                complete = False
                        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                              and isinstance(node.func.value, ast.Name) and node.func.value.id == argument
                              and node.func.attr == 'get'):
                            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                keys.add(node.args[0].value)
                            else:
                                complete = False
                    schema_inputs = {'type': 'object', 'properties': {key: {} for key in sorted(keys)},
                                     'required': sorted(required_keys), 'additionalProperties': True}
                    basis = 'admitted_source_public_input_keys'
                # This reserved control object is the actual executor calling
                # convention, not an example with a hidden correct answer.
                parameters = copy.deepcopy(schema_inputs)
                parameters.setdefault('properties', {})['control'] = copy.deepcopy(schema_inputs)
                operations[action] = {
                    'source': 'admitted', 'operation': action,
                    'observable': str(receipt.get('observable') or ''),
                    'metric': str(receipt.get('observable') or ''),
                    'parameters_schema': parameters, 'schema_basis': basis,
                    'static_input_keys_complete': complete,
                    'value_domains': 'undeclared' if basis.endswith('public_input_keys') else 'receipt_schema',
                    'control_semantics': {'field': 'parameters.control',
                        'treatment': 'parameters excluding control',
                        'control': 'parameters.control passed to the same act(params) function',
                        'omitted': 'same_parameters_as_treatment', 'empty_object': 'same_parameters_as_treatment'},
                    'implementation_digest': receipt.get('implementation_digest'),
                }
            except (OSError, ValueError, SyntaxError, StopIteration, IndexError, TypeError):
                pass  # Missing/unverifiable interface is not a callable promise.
    metrics = tuple(sorted({metric for api in operations.values()
                            for metric in api.get('metrics', [api.get('metric')]) if metric}))
    return {
        "world_id": str(getattr(fixture, "id", "") or ""),
        "world_class": world_class,
        "schema": schema,
        "callable_operations": operations,
        "interfaces_attested": attested,
        "schema_declared_capabilities": caps,
        "observations": metrics,
        "interventions": tuple(sorted(operations)),
        "reset_semantics": "rebind attested bytes",
        "isolation": "workspace copy",
        "determinism": True,
        "datasets": list(getattr(fixture, "files", ()) or ()),
        "executables": tuple(sorted(operations)),
        "tools": tuple(sorted(operations)),
        "metrics": metrics,
        "compute": "local_stdlib",
        "filesystem_access": "workspace_only",
        "network_access": False,
        "state_persistence": False,
        "provenance": str(getattr(fixture, "provenance", "") or "catalog"),
        "attestation": "byte_digest",
        "cost": "cheap",
        "replication_support": True,
        "contamination_risk": "low",
        "limitations": (
            "fixture capabilities are schema-declared; they are not a measured "
            "dynamics report"
        ),
    }


def missing_experiment_needs(req: ExperimentRequirement) -> dict[str, bool]:
    """Capability gap as missing experimental parts, not missing_operator=X."""
    from .experiment import OPERATIONS

    execution = dict(req.execution or {})
    extension = dict(req.extension or {})
    op = execution.get("operation")
    has_action = (isinstance(op, str) and op in OPERATIONS) or bool(extension.get("action"))
    has_observable = bool(execution.get("metric") or extension.get("observable"))
    has_evaluator = bool(execution.get("expected_direction") or extension.get("evaluator"))
    has_instrument = bool(extension.get("implementation") or has_action)
    return {
        "missing_intervention": not has_action,
        "missing_observable": not has_observable,
        "missing_evaluator": not has_evaluator,
        "missing_asset": not bool(req.world_class),
        "missing_control": not has_instrument,
        "missing_resource": bool(req.resource_requirements.get("infeasible")),
    }


def run_contract(
    req: ExperimentRequirement,
    fixture: WorldFixture,
    dest: Path,
    *,
    study_stage: str = PILOT,
) -> dict[str, Any]:
    """One execution entry. Stage is evidence semantics, not a second architecture."""
    return run_computational_stage(req, fixture, dest, study_stage=study_stage)


def _unsupported_reason(req: ExperimentRequirement) -> str:
    if _WETLAB.search(req.hypothesis + " " + req.measurement):
        return "requires_physical_or_wet_lab_measurement"
    if not req.world_class:
        return "no_supported_computational_world_class"
    return "required_capability_unavailable"


def classify_world_match(
    req: ExperimentRequirement,
    *,
    catalog: Mapping[str, WorldFixture] | None = None,
    bound: WorldFixture | None = None,
    catalog_dir: Path | None = None,
) -> dict[str, Any]:
    """Richer than exists/missing. World answers executability only."""
    if req.resource_requirements.get("infeasible"):
        return {
            "status": RESOURCE_BLOCKED,
            "reason": str(req.resource_requirements.get("reason") or "resource_infeasible"),
            "missing_capability": "",
            "missing_resource": "gpu_or_private_data",
            "why_generated_cannot_replace_evidence": (
                "a model rollout is GENERATED and cannot stand in for the declared "
                "measurement"
            ),
            "world_id": "",
            "world_class": req.world_class,
        }
    from .experiment import protocol_errors, fixture_errors, admitted_receipt
    if bound is not None:
        receipt = admitted_receipt(bound)
        wanted = str((req.execution or {}).get('operation') or (req.extension or {}).get('action') or '')
        if receipt.get('admitted') and wanted and receipt.get('action') == wanted:
            errors = fixture_errors(req, bound)
            if errors:
                return {'status': MATCH_REQUIRES_INSTRUMENTATION, 'reason': '; '.join(errors),
                        'missing_capability': '; '.join(errors), 'missing_resource': '',
                        'world_id': bound.id, 'world_class': req.world_class,
                        'why_generated_cannot_replace_evidence': 'the bound interface must pass attestation and protocol validation'}
            return {'status': MATCHED_EXISTING_WORLD, 'reason': 'admitted_operation_matches_attested_interface',
                    'world_id': bound.id, 'world_class': req.world_class,
                    'evidence_scope': 'registered_operation_on_attested_world',
                    'scientific_claim_supported': False}
    if not req.world_class or req.world_class not in WORLD_CLASSES:
        return {
            "status": UNSUPPORTED_WORLD,
            "reason": _unsupported_reason(req),
            "missing_capability": req.world_class or "computational_world",
            "missing_resource": "",
            "why_generated_cannot_replace_evidence": (
                "OpenWorld simulation and critic judgement are GENERATED; they do "
                "not measure the physical or unavailable object the claim names"
            ),
            "world_id": "",
            "world_class": req.world_class,
        }
    def blocked(reason: str, world_id: str = "") -> dict[str, Any]:
        return {"status": MATCH_REQUIRES_INSTRUMENTATION, "reason": reason,
                "missing_capability": reason, "missing_resource": "", "world_id": world_id,
                "world_class": req.world_class,
                "why_generated_cannot_replace_evidence":
                    "a sandbox cannot replace the registered experiment or its required scale"}
    blob = " ".join((req.hypothesis, req.measurement, req.intervention,
                     " ".join(req.required_external_assets)))
    if _TEMPLATE_TOO_SMALL.search(blob):
        return blocked("operator_template_cannot_answer_this_claim_scale")
    errors = protocol_errors(req)
    gap_marks = {"missing_executable_protocol", "unsupported_experiment_operator",
                 "world_extension_awaits_admission"}
    admitted_bound = bound is not None and not fixture_errors(req, bound)
    if (errors and gap_marks.intersection(errors) and not _WETLAB.search(blob)
            and not admitted_bound):
        missing = missing_experiment_needs(req)
        return {
            "status": CAPABILITY_GAP,
            "reason": "; ".join(errors),
            "missing_capability": "registered_or_admitted_executor",
            "missing_resource": "",
            "missing": missing,
            "world_id": "",
            "world_class": req.world_class,
            "why_generated_cannot_replace_evidence": (
                "a model-written experiment is a candidate capability; it cannot "
                "certify itself as a WORLD executor"
            ),
        }
    if errors and not admitted_bound:
        return blocked("; ".join(errors))
    supported = {
        REPOSITORY: {"repository", "checkout", "files", "tests", "commands", "reset"},
        DATASET: {"table", "rows", "numeric", "split", "metric", "reset"},
        SIMULATION: {"graph", "nodes", "edges", "seed", "repeat", "reset"},
    }.get(req.world_class, set())
    if set(req.required_world_capabilities) - supported:
        return blocked("required_world_capability_not_implemented")
    worlds = dict(catalog or {})
    if catalog_dir is not None:
        from .recipes import catalog_worlds
        worlds.update({w.id: w for w in catalog_worlds(Path(catalog_dir))})
    if bound is not None:
        errors = fixture_errors(req, bound)
        if errors:
            return blocked("; ".join(errors), bound.id)
        return {"status": MATCHED_EXISTING_WORLD, "reason": "registered_operator_matches_attested_payload",
                "world_id": bound.id, "world_class": req.world_class}
    for fixture in worlds.values():
        if not fixture_errors(req, fixture):
            return {"status": CAN_ACQUIRE_WORLD, "reason": "registered_operator_matches_catalog_payload",
                    "world_id": fixture.id, "world_class": req.world_class}
    # Validate actual template payload as well: operator-specific applicability,
    # available independent samples and filename/column requirements all matter.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="farfield-world-match-") as tmp:
        template = build_template_world(req.world_class, Path(tmp))
        errors = fixture_errors(req, template)
    if errors:
        return blocked("; ".join(errors))
    return {"status": CAN_BUILD_WORLD, "reason": "registered_operator_matches_template_payload",
            "world_id": "", "world_class": req.world_class, "recipe": CONSTRUCT_TEMPLATE}


def protocol_runnable(
    req: ExperimentRequirement,
    match: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Can this protocol be executed? Not: is the idea worth pursuing?"""
    from .experiment import protocol_errors

    classified = dict(match) if match is not None else classify_world_match(req)
    if str(classified.get("status") or "") == MATCHED_EXISTING_WORLD:
        return {"runnable": True, "reason": "matched_existing_world", "errors": [],
                "match": classified}
    errors = protocol_errors(req)
    if errors:
        return {"runnable": False, "reason": "; ".join(errors), "errors": list(errors)}
    if req.resource_requirements.get("infeasible"):
        return {"runnable": False, "reason": "resource_infeasible", "errors": ["resource_infeasible"]}
    status = str(classified.get("status") or "")
    if status in {RESOURCE_BLOCKED, UNSUPPORTED_WORLD, MATCH_REQUIRES_INSTRUMENTATION,
                  CAPABILITY_GAP}:
        return {"runnable": False, "reason": str(classified.get("reason") or status),
                "errors": [status], "match": classified}
    return {"runnable": True, "reason": status or "protocol_valid", "errors": [],
            "match": classified}


def research_promise(feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    """Worth pursuing as research? Not: may the protocol run?

    Rank string is the discovery function. This wrapper does not invent a
    second HIGH/LOW rule.
    """
    from .discovery import COLLAPSE, PROMISING, REVISE
    from .discovery import research_promise as promise_rank

    fb = dict(feedback or {})
    verdict = str(fb.get("verdict") or "")
    return {
        "verdict": verdict,
        "promising": verdict == PROMISING,
        "revise": verdict == REVISE,
        "collapse": verdict == COLLAPSE,
        "root_failure_cause": str(fb.get("root_failure_cause") or ""),
        "research_promise": promise_rank(fb),
    }


def cheap_execution_stage(feedback: Mapping[str, Any] | None) -> str:
    """Calibration vs scientific cheap run. Critic verdict is not rewritten."""
    return SCIENTIFIC_PILOT if research_promise(feedback)["promising"] else CALIBRATION


def execution_feedback(
    req: ExperimentRequirement,
    match: Mapping[str, Any],
    *,
    pilot_verdict: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Structured signal Discovery can consume. Not evidence."""
    status = str(match.get("status") or "")
    failure = str(pilot_verdict or status)
    operator = FAILURE_TO_OPERATOR.get(failure, FAILURE_TO_OPERATOR.get(status, "REFORMULATE"))
    repairable = status not in {UNSUPPORTED_WORLD} or bool(req.world_class)
    if status == UNSUPPORTED_WORLD and _WETLAB.search(req.hypothesis):
        repairable = False
    from .discovery import FAILURE_CHANGES, classify_research_failure

    failure_class = classify_research_failure(
        match=match, pilot_verdict=pilot_verdict, cause=failure, repairable=repairable)
    return {
        "kind": "execution_feedback",
        "epistemic": "GENERATED",
        "failure_type": failure,
        "failure_class": failure_class,
        "must_change": "" if pilot_verdict == PILOT_NEGATIVE else FAILURE_CHANGES.get(failure_class, ""),
        "required_resource": match.get("missing_resource") or "",
        "missing_capability": match.get("missing_capability") or "",
        "repairable": repairable,
        "suggested_operator": operator,
        "detail": detail or str(match.get("reason") or ""),
        "why_generated_cannot_replace_evidence": match.get(
            "why_generated_cannot_replace_evidence") or "",
        "claim_id": req.claim_id,
        "world_class": req.world_class,
    }


def suggested_operator_for_execution(failure_type: str) -> str:
    return FAILURE_TO_OPERATOR.get(str(failure_type or ""), "REFORMULATE")


# ---------------------------------------------------------------------------
# Template worlds (operator-owned). Not model-generated environments.
# ---------------------------------------------------------------------------


def _write_fixture(dest: Path, *, world_id: str, schema: str, world_class: str,
                   title: str, files: dict[str, str], domains: tuple[str, ...],
                   source: str = "") -> WorldFixture:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = sorted(files)
    for name, text in files.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    digest = digest_files(dest, names)
    manifest = {
        "id": world_id,
        "title": title,
        "source": source or TEMPLATE_SOURCE[world_class],
        "retrieved_at": date.today().isoformat(),
        "digest": digest,
        "files": names,
        "domains": list(domains),
        "role": "world",
        "schema": schema,
        "provenance": "constructed",
        "world_class": world_class,
        "capability_attestation": None,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load_fixture(dest)


def build_template_world(world_class: str, dest: Path, *, world_id: str = "") -> WorldFixture:
    """Construct a tiny deterministic sandbox for one computational class."""
    if world_class not in WORLD_CLASSES:
        raise ValueError(f"no template for world class {world_class!r}")
    wid = world_id or f"template-{world_class}"
    if world_class == REPOSITORY:
        files = {
            "repo.py": (
                "VALUE = 1\n"
                "def score():\n"
                "    return VALUE\n"
            ),
            "test_repo.py": (
                "import unittest\n"
                "import repo\n"
                "class T(unittest.TestCase):\n"
                "    def test_score(self):\n"
                "        self.assertEqual(repo.score(), 2)\n"
            ),
        }
        return _write_fixture(
            dest, world_id=wid, schema="repository", world_class=REPOSITORY,
            title="Minimal repository sandbox", files=files,
            domains=("repository", "software", "tests"))
    if world_class == DATASET:
        rows = [{"x": i, "y": int(i >= 10), "z": i * 0.1} for i in range(20)]
        files = {
            "table.json": json.dumps({"columns": ["x", "y", "z"], "rows": rows}, indent=2) + "\n",
        }
        return _write_fixture(
            dest, world_id=wid, schema="numeric_table", world_class=DATASET,
            title="Minimal numeric table", files=files,
            domains=("table", "numeric", "benchmark"))
    if world_class == SIMULATION:
        files = {
            "graph.json": json.dumps({
                "nodes": [0, 1, 2, 3, 4, 5],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            }, indent=2) + "\n",
        }
        return _write_fixture(
            dest, world_id=wid, schema="undirected_graph", world_class=SIMULATION,
            title="Minimal cycle graph", files=files,
            domains=("graph", "network", "simulation"))
    files = {
        "job.json": json.dumps({
            "model": "fake-local",
            "prompts": ["alpha", "beta", "gamma", "delta"],
            "seeds": [0, 1],
            "temperatures": [0.0, 0.8],
        }, indent=2) + "\n",
    }
    return _write_fixture(
        dest, world_id=wid, schema="inference_job", world_class=INFERENCE,
        title="Fake inference job", files=files,
        domains=("inference", "decode", "seed"))


def attest_world_capabilities(fixture: WorldFixture, req: ExperimentRequirement) -> dict[str, Any]:
    """Prove declared capabilities exist. Byte digest is necessary, not sufficient."""
    from .experiment import fixture_errors
    errors = fixture_errors(req, fixture)
    record = {"ok": not errors, "checks": {"registered_operator_and_payload": not errors},
              "errors": errors, "world_id": fixture.id, "digest": fixture.digest,
              "world_class": req.world_class, "attested_for_confirmatory": not errors}
    return record


def admit_research_capability(fixture: WorldFixture, req: ExperimentRequirement) -> dict[str, Any]:
    """A generated world cannot admit itself. Only attested payloads pass."""
    from .worldio import is_attested

    provenance = str(getattr(fixture, "provenance", "") or "")
    if provenance == "generated":
        return {"admitted": False, "status": "candidate",
                "reason": "generated_capability_cannot_self_attest"}
    record = attest_world_capabilities(fixture, req)
    admitted = bool(record.get("ok") and (
        is_attested(fixture) or provenance == "constructed"))
    return {"admitted": admitted, "status": "admitted" if admitted else "candidate",
            "reason": "" if admitted else "; ".join(record.get("errors") or ["not_attested"]),
            "attestation": record}


def world_surface(state: Any) -> dict[str, Any]:
    """Derived view: registered operators plus attested admitted actions."""
    from .experiment import OPERATIONS, admitted_receipt

    actions: dict[str, Any] = {
        name: {"source": "registered", "world_class": spec[0]}
        for name, spec in OPERATIONS.items()
    }
    versions = list(getattr(state, "world_versions", []) or [])
    if isinstance(state, Mapping):
        versions = list(state.get("world_versions") or versions)
    for row in versions:
        if row.get("deprecated") or not row.get("path"):
            continue
        try:
            fixture = load_fixture(Path(row["path"]))
        except (OSError, ValueError, KeyError):
            continue
        receipt = admitted_receipt(fixture)
        action = str(receipt.get("action") or "")
        if not (receipt.get("admitted") and action):
            continue
        actions[action] = {
            "source": "admitted",
            "world_id": fixture.id,
            "digest": fixture.digest,
            "observable": receipt.get("observable") or "",
        }
    return {"actions": actions}


def admit_world_extension(fixture: WorldFixture, req: ExperimentRequirement) -> dict[str, Any]:
    """Agent proposes an implementation. Kernel decides against independent controls.

    Proposer tests, if present, are ignored. Passing them cannot admit.
    """
    from .experiment import (
        digest,
        implementation_errors,
        independent_controls,
        run_independent_controls,
    )
    from .worldio import is_attested

    provenance = str(getattr(fixture, "provenance", "") or "")
    if provenance == "generated":
        return {"admitted": False, "status": "UNADMITTED",
                "reason": "generated_capability_cannot_self_attest"}
    if not is_attested(fixture):
        return {"admitted": False, "status": "UNADMITTED", "reason": "unattested_world"}
    controls = independent_controls(fixture)
    if not controls:
        return {"admitted": False, "status": "INSTRUMENT_INVALID",
                "reason": "missing_independent_controls"}
    wanted = str((req.extension or {}).get("action") or "")
    named = str(controls.get("action") or "")
    if wanted and named and wanted != named:
        return {"admitted": False, "status": "UNADMITTED", "reason": "extension_action_mismatch"}
    source = str((req.extension or {}).get("implementation") or "")
    errors = implementation_errors(source)
    if errors:
        return {"admitted": False, "status": "UNADMITTED", "reason": ";".join(errors)}
    first = run_independent_controls(source, controls)
    if not first.get("ok"):
        return {"admitted": False,
                "status": str(first.get("status") or "INSTRUMENT_INVALID"),
                "reason": str(first.get("reason") or "independent_controls_failed"),
                "proposer_tests_ignored": True}
    second = run_independent_controls(source, controls)
    if second.get("digest") != first.get("digest"):
        return {"admitted": False, "status": "UNADMITTED", "reason": "not_reproducible",
                "proposer_tests_ignored": True}
    action = named or wanted
    return {
        "admitted": True,
        "status": "admitted",
        "reason": "",
        "action": action,
        "observable": controls.get("observable") or "",
        "evaluator": controls.get("evaluator") or "exact_match",
        "implementation_digest": digest(source),
        "controls_digest": digest(controls),
        "replay_digest": first.get("digest"),
        "proposer_tests_ignored": True,
        "parent_id": fixture.id,
        "parent_digest": fixture.digest,
    }


def build_admitted_world(
    parent: WorldFixture,
    dest: Path,
    source: str,
    receipt: Mapping[str, Any],
    *,
    world_id: str = "",
) -> WorldFixture:
    """Child world. Writing into the parent would change its digest."""
    action = str(receipt.get("action") or "action")
    files = {name: (parent.root / name).read_text(encoding="utf-8") for name in parent.files}
    files[f"admitted/{action}.py"] = source if source.endswith("\n") else source + "\n"
    files["admitted_receipt.json"] = json.dumps(dict(receipt), sort_keys=True, indent=2) + "\n"
    return _write_fixture(
        dest,
        world_id=world_id or f"admitted-{action}",
        schema=parent.schema or CLASS_SCHEMA[SIMULATION],
        world_class=SIMULATION,
        title=f"Admitted {action}",
        files=files,
        domains=tuple(parent.domains) or ("computational",),
        source=f"farfield-admitted:{parent.id}",
    )


def build_control_world(
    dest: Path,
    controls: Mapping[str, Any],
    *,
    world_id: str = "independent-controls",
) -> WorldFixture:
    """Attested substrate that holds kernel-owned acceptance cases. Not a proposer file."""
    files = {
        "independent_controls.json": json.dumps(dict(controls), indent=2) + "\n",
        "substrate.json": json.dumps({"kind": "computational_substrate"}, indent=2) + "\n",
    }
    return _write_fixture(
        dest, world_id=world_id, schema=CLASS_SCHEMA[SIMULATION], world_class=SIMULATION,
        title="Independent control substrate", files=files,
        domains=("computational", "controls"))


def can_support_confirmatory(fixture: WorldFixture, req: ExperimentRequirement) -> bool:
    """Candidate worlds (model-generated, unattested) cannot confirm."""
    return bool(admit_research_capability(fixture, req).get("admitted"))


def capability_gaps(state: Any) -> list[dict[str, Any]]:
    """Derived CAPABILITY_GAP rows. Not an admission, not a world."""
    from .discovery import EMPIRICAL_CLAIMS, discovery_view, idea_of

    view = discovery_view(state)
    found: list[dict[str, Any]] = []
    for row in view["ideas"]:
        if idea_of(row).claim_type not in EMPIRICAL_CLAIMS:
            continue
        req = requirement_of_idea(state, row)
        match = classify_world_match(req)
        if match.get("status") != CAPABILITY_GAP:
            continue
        found.append({
            "idea_id": str(row.get("id") or ""),
            "world_class": req.world_class,
            "missing_capability": match.get("missing_capability") or "",
            "missing": dict(match.get("missing") or missing_experiment_needs(req)),
            "reason": match.get("reason") or "",
            "match": match,
        })
    return found


# ---------------------------------------------------------------------------
# Pilot execution (V3 cheap external). Real stdlib run, not a simulation.
# ---------------------------------------------------------------------------


def _pilot_script(world_class: str) -> str:
    # All operators read the exact protocol written before process launch.
    return (
        "import json, sys\nfrom pathlib import Path\n"
        + "sys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[3])) + ")\n"
        "from farfield.extras.openworld.experiment import execute_protocol\n"
        "doc = json.loads(Path('protocol.json').read_text())\n"
        "out = execute_protocol(doc['registration'], Path('data'), Path('work'),\n"
        "                       setting_id=doc.get('study_setting_id', ''))\n"
        "Path('metrics.json').write_text(json.dumps(out, allow_nan=False))\n"
    )


def judge_pilot(req: ExperimentRequirement, metrics: Mapping[str, Any]) -> str:
    from ..validity import effect_verdict, separation
    verdict = effect_verdict(separation(metrics.get('treatment'), metrics.get('control')),
                             req.execution.get('margin'), req.execution.get('expected_direction'))
    return {'supports': PILOT_POSITIVE, 'weakens': PILOT_NEGATIVE}.get(verdict, PILOT_INCONCLUSIVE)


def run_computational_stage(req: ExperimentRequirement, fixture: WorldFixture, dest: Path,
                            *, study_stage: str = PILOT, workspace_root: Path | None = None,
                            setting_id: str = "") -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    from .experiment import fixture_errors, registration, digest, safe_identifier, confined_path
    dest = Path(dest).absolute()
    errors = []
    if not safe_identifier(req.claim_id):
        errors.append('invalid_computational_claim_id')
    try:
        confined_path(workspace_root or dest.parent, str(dest))
        for name in ('protocol.json', 'metrics.json', 'experiment.py', 'data', 'work'):
            confined_path(dest, name)
        for name in fixture.files:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('world_artifact_escapes_freeze')
            confined_path(fixture.root, name)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        return {'ok': False, 'execution_status': 'blocked', 'reason': '; '.join(errors),
                'pilot_verdict': PILOT_INCONCLUSIVE, 'metrics': {}, 'study_stage': study_stage}
    errors = fixture_errors(req, fixture)
    if study_stage not in STUDY_STAGES:
        errors.append('invalid_study_stage')
    if errors:
        return {'ok': False, 'execution_status': 'blocked', 'reason': '; '.join(errors),
                'pilot_verdict': PILOT_INCONCLUSIVE, 'metrics': {}, 'study_stage': study_stage}
    dest.mkdir(parents=True, exist_ok=True)
    reg = registration(req)
    from .study import setting_ids, study_of
    if study_of(req.execution) is not None and setting_id not in setting_ids(req.execution):
        return {'ok': False, 'execution_status': 'blocked',
                'reason': 'unregistered_study_setting', 'pilot_verdict': PILOT_INCONCLUSIVE,
                'metrics': {}, 'study_stage': study_stage}
    protocol = {'registration': reg, 'registration_digest': digest(reg),
                'world_id': fixture.id, 'world_digest': fixture.digest,
                'study_stage': study_stage, 'study_setting_id': str(setting_id or '')}
    protocol_path = dest / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        errors.append('registered_protocol_changed')
    if errors:
        return {'ok': False, 'execution_status': 'blocked', 'reason': '; '.join(errors),
                'pilot_verdict': PILOT_INCONCLUSIVE, 'metrics': {}, 'study_stage': study_stage}
    protocol_path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + '\n')
    for name in ('data', 'work'):
        if (dest / name).exists():
            shutil.rmtree(dest / name)
    (dest / 'metrics.json').unlink(missing_ok=True)
    (dest / 'data').mkdir()
    for name in fixture.files:
        source = fixture.root / name
        if not source.resolve().is_relative_to(fixture.root.resolve()):
            raise ValueError('world_artifact_escapes_freeze')
        target = dest / 'data' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if digest_files(dest / 'data', fixture.files) != fixture.digest:
        return {'ok': False, 'execution_status': 'blocked', 'reason': 'copied_world_digest_mismatch',
                'pilot_verdict': PILOT_INCONCLUSIVE, 'metrics': {}, 'study_stage': study_stage}
    script = _pilot_script(req.world_class)
    (dest / 'experiment.py').write_text(script)
    # Preserve only safe runtime settings. Provider credentials never reach tests.
    env = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        proc = subprocess.run([sys.executable, 'experiment.py'], cwd=dest, env=env,
                              capture_output=True, text=True, timeout=30, check=False)
        metrics_path = dest / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if proc.returncode == 0 and metrics_path.is_file() else {}
        ok = proc.returncode == 0 and bool(metrics)
        if (digest_files(dest / 'data', fixture.files) != fixture.digest
                or json.loads(protocol_path.read_text()) != protocol
                or (dest / 'experiment.py').read_text() != script):
            raise ValueError('executed_artifacts_changed')
        result = {'ok': ok, 'execution_status': 'ran' if ok else 'failed',
                  'reason': '' if ok else 'experiment_process_failed', 'metrics': metrics,
                  'stdout': proc.stdout[-500:], 'stderr': proc.stderr[-500:], 'returncode': proc.returncode}
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        result = {'ok': False, 'execution_status': 'failed', 'reason': type(exc).__name__, 'metrics': {}}
    result.update(pilot_verdict=judge_pilot(req, result['metrics']) if result['ok'] else PILOT_INCONCLUSIVE,
                  study_stage=study_stage, world_id=fixture.id, world_digest=fixture.digest,
                  registration=reg, registration_digest=digest(reg), reads_data=bool(result['ok']),
                  study_setting_id=str(setting_id or ''))
    return result


def run_computational_pilot(
    req: ExperimentRequirement,
    fixture: WorldFixture,
    dest: Path,
) -> dict[str, Any]:
    return run_computational_stage(req, fixture, dest, study_stage=PILOT)


def stage_evidence_row(
    req: ExperimentRequirement,
    fixture: WorldFixture,
    run: Mapping[str, Any],
    *,
    question_id: str = "",
    action_type: str = "PROBE",
    study_stage: str = PILOT,
    theory_id: str = "",
    replication_of: str = "",
    setting_id: str = "",
) -> dict[str, Any]:
    """Trusted observation of a cheap run. Pilot cannot support the central claim."""
    row = pilot_evidence_row(
        req, fixture, run, question_id=question_id, action_type=action_type)
    stage = study_stage if study_stage in STUDY_STAGES else PILOT
    row["study_stage"] = stage
    row["theory_id"] = theory_id
    if setting_id:
        # Which registered setting this reading covers. Coverage is counted from
        # these ids; it is not the same count as independent units.
        row["study_setting_id"] = str(setting_id)
    if replication_of:
        row["replication_of"] = replication_of
    if stage in CHEAP_STAGES:
        row["promotion_ok"] = False
        row["cannot_corroborate"] = True
        row["note"] = (
            "calibration observation may be WORLD; it cannot raise central_claim_support"
            if stage == CALIBRATION else
            "pilot observation may be WORLD; it cannot raise central_claim_support"
        )
        if stage == CALIBRATION:
            row["execution_kind"] = CALIBRATION
            row["protocol_scope"] = "calibration_fixture_verification_only"
    else:
        row["promotion_ok"] = False
        row["note"] = (
            "confirmatory/replication observation may be WORLD; validity still "
            "decides whether it can corroborate a claim"
        )
        if run.get("pilot_verdict") == PILOT_POSITIVE:
            row["stage_verdict"] = "POSITIVE"
        elif run.get("pilot_verdict") == PILOT_NEGATIVE:
            row["stage_verdict"] = "NEGATIVE"
        else:
            row["stage_verdict"] = "INCONCLUSIVE"
    from .experiment import claim_spec, inference_link, registration, digest
    relation = {'PILOT_POSITIVE': 'supports', 'PILOT_NEGATIVE': 'contradict'}.get(
        str(run.get('pilot_verdict') or ''), 'inconclusive')
    row['inference_link'] = inference_link(req, study_stage=stage, relation=relation)
    if run.get('ok'):
        if (run.get('registration') != registration(req)
                or run.get('registration_digest') != digest(registration(req))
                or run.get('world_id') != fixture.id or run.get('world_digest') != fixture.digest
                or run.get('study_stage') != stage
                or str(run.get('study_setting_id') or '') != str(setting_id or '')):
            raise ValueError('successful_run_does_not_match_registered_claim_world_or_stage')
        row.update(
            margin=req.execution['margin'], expected_direction=req.execution['expected_direction'],
            sample_size=run['metrics']['sample_size'], sample_size_basis=run['metrics']['sample_size_basis'],
            card={'claim_spec': claim_spec(req, fixture), 'world_observable': req.primary_metric},
            diagnosis={'world_observable': req.primary_metric},
            registration=run['registration'], registration_digest=run['registration_digest'],
            validation_gaps=[],
            # These operators directly manipulate the measured object or run a
            # predictive holdout comparison; neither claims a hidden mechanism.
            oracle_required=False, placebo_required=False, ablation_required=False,
            protocol_scope='registered_operation_on_fixed_fixture_only',
        )
    else:
        row['validation_gaps'] = [str(run.get('reason') or 'experiment_did_not_run')]
    return row


def pilot_evidence_row(
    req: ExperimentRequirement,
    fixture: WorldFixture,
    run: Mapping[str, Any],
    *,
    question_id: str = "",
    action_type: str = "PROBE",
) -> dict[str, Any]:
    """Trusted observation of a cheap run. Does not support the central claim."""
    metrics = dict(run.get("metrics") or {})
    treatment = metrics.get("treatment")
    control = metrics.get("control")
    from .experiment import digest
    identity = {'claim_id': req.claim_id, 'world_digest': fixture.digest,
                'registration_digest': run.get('registration_digest'),
                'metrics': metrics, 'stage': run.get('study_stage') or PILOT}
    if run.get('study_setting_id'):
        # Two settings can measure the same numbers and still be two readings.
        # The key is added only when there is one, so every identity recorded
        # before studies existed keeps the bytes it was recorded under.
        identity['study_setting_id'] = str(run['study_setting_id'])
    eid = 'obs-' + digest(identity)[:24]
    return {
        "evidence_id": eid,
        "role": "ProbeEvidence",
        "study_stage": PILOT,
        "pilot_verdict": run.get("pilot_verdict"),
        "discovery_idea_id": req.claim_id,
        "question_id": question_id,
        "theory_id": "",
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "experiment_digest": hashlib.sha256(_pilot_script(req.world_class).encode()).hexdigest(),
        "epistemic": "WORLD" if run.get("ok") else "GENERATED",
        "attested": bool(run.get("ok")),
        "promotion_ok": False,
        "cannot_corroborate": True,
        "corroborates": False,
        "outcome": "inconclusive",
        "verdict": "uninformative",
        "treatment": treatment,
        "control": control,
        "n_treatment": metrics.get("n_treatment"),
        "n_control": metrics.get("n_control"),
        "primary_metric": req.primary_metric,
        "metrics": metrics,
        "execution_status": "ran" if run.get("ok") else "blocked",
        "reads_world_data": True,
        "note": "pilot observation may be WORLD; it cannot raise central_claim_support",
        "action_type": action_type,
    }


def replication_record(*, available: bool, reason: str = "") -> dict[str, Any]:
    if available:
        return {"status": "REPLICATION_AVAILABLE", "reason": reason}
    return {"status": "REPLICATION_UNAVAILABLE", "reason": reason or "no_compatible_heldout"}


# ---------------------------------------------------------------------------
# Exploration evaluation interface (no large benchmark).
# ---------------------------------------------------------------------------


def exploration_trial_record(
    *,
    arm: str,
    idea_id: str = "",
    novelty: str = "",
    mechanistic_coherence: str = "",
    falsifiability: str = "",
    validation_survived: bool = False,
    pilot_survived: bool = False,
    cost: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if arm not in EXPLORATION_ARMS:
        raise ValueError(f"unknown exploration arm {arm!r}")
    return {
        "arm": arm,
        "idea_id": idea_id,
        "novelty": novelty,
        "mechanistic_coherence": mechanistic_coherence,
        "falsifiability": falsifiability,
        "validation_survived": bool(validation_survived),
        "pilot_survived": bool(pilot_survived),
        "cost": dict(cost or {}),
        "ranking_signal": "downstream_validation_and_pilot_not_distance",
        "exploration_advantage": "NOT_YET_VALIDATED",
    }


def matched_budget_comparison(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Best validated idea @ fixed compute. Distance is not a score."""
    by_arm: dict[str, list[dict[str, Any]]] = {name: [] for name in EXPLORATION_ARMS}
    for row in rows:
        arm = str(row.get("arm") or "")
        if arm in by_arm:
            by_arm[arm].append(dict(row))
    scored = []
    for arm, items in by_arm.items():
        survived = sum(1 for item in items if item.get("pilot_survived")
                       or item.get("validation_survived"))
        scored.append({"arm": arm, "n": len(items), "survived": survived})
    scored.sort(key=lambda item: (-item["survived"], item["arm"]))
    return {
        "metric": "best_validated_idea_at_fixed_compute",
        "arms": scored,
        "winner": scored[0]["arm"] if scored and scored[0]["survived"] else "",
        "exploration_advantage": "NOT_YET_VALIDATED",
        "distance_used_as_score": False,
    }


def operator_downstream_record(
    *,
    research_state: str,
    operator: str,
    next_idea_id: str,
    validation: str,
    pilot: str = "",
    cost: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Did this operator leave a better subsequent research state?"""
    improved = validation in {"PROMISING", "PILOT_POSITIVE"} or pilot == PILOT_POSITIVE
    return {
        "research_state_t": research_state,
        "operator_t": operator,
        "idea_t1": next_idea_id,
        "validation_t1": validation,
        "pilot_t1": pilot,
        "cost": dict(cost or {}),
        "improved": improved,
        "credit_basis": "downstream_outcome_not_operator_use",
    }


def requirement_of_idea(state: Any, idea_row: Mapping[str, Any]) -> ExperimentRequirement:
    """Compile the contract for one stored idea row."""
    from .discovery import idea_of, root_question, discovery_view

    view = discovery_view(state)
    feedback = view["feedback"].get(str(idea_row.get("id") or ""))
    return compile_experiment_requirement(
        idea_of(idea_row),
        idea_id=str(idea_row.get("id") or ""),
        question_id=str(idea_row.get('question_id') or 'Q1'),
        feedback=feedback,
        root_question=root_question(state),
    )


def acquired_world_id(state: Any, req: ExperimentRequirement) -> str:
    """A constructed or attested world already bound for this contract, or ""."""
    versions = list(getattr(state, "world_versions", []) or [])
    if isinstance(state, Mapping):
        versions = list(state.get("world_versions") or versions)
    from .experiment import adaptive_criterion_sources
    adaptive = bool(adaptive_criterion_sources(req.execution, state))
    current = state.get('world_id') if isinstance(state, Mapping) else state.world_id
    # A revised criterion has an explicitly bound evaluation world. Selecting
    # the old training fixture here would defeat the execution layer's guard.
    if adaptive:
        versions = sorted(versions, key=lambda w: w.get('id') != current)
    for row in versions:
        if (row.get('research_role') == 'heldout' and not (adaptive and row.get('id') == current)) or not row.get('path'):
            continue
        try:
            fixture = load_fixture(Path(row['path']))
            if classify_world_match(req, bound=fixture)['status'] == MATCHED_EXISTING_WORLD:
                return fixture.id
        except (OSError, ValueError, KeyError):
            continue
    return ""


def bound_fixture(state: Any, workspace: Path | None = None) -> WorldFixture | None:
    from .discovery import evidence_world_bound

    if not evidence_world_bound(state) or workspace is None:
        return None
    wid = str(getattr(state, "world_id", "") or "")
    if isinstance(state, Mapping):
        wid = str(state.get("world_id") or wid)
    if not wid:
        return None
    root = Path(workspace) / "worlds" / wid
    if not (root / "manifest.json").is_file():
        return None
    try:
        return load_fixture(root)
    except (OSError, ValueError, KeyError):
        return None


def match_of_idea(
    state: Any,
    idea_row: Mapping[str, Any],
    *,
    catalog_dir: Path | None = None,
    bound: WorldFixture | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    req = requirement_of_idea(state, idea_row)
    if bound is None and workspace is not None:
        bound = bound_fixture(state, workspace)
    if bound is None:
        # Discovery and the execution scheduler must classify the same registered
        # instrument. The canonical world registry also exists without a worker
        # workspace argument; absence of that argument is not a capability gap.
        wid = acquired_world_id(state, req)
        versions = (state.get('world_versions', []) if isinstance(state, Mapping)
                    else getattr(state, 'world_versions', []))
        version = next((w for w in versions if (w.get('id') or w.get('world_id')) == wid), {})
        if wid and version.get('path'):
            try:
                bound = load_fixture(Path(version['path']))
            except (OSError, ValueError, KeyError):
                pass
    return classify_world_match(req, bound=bound, catalog_dir=catalog_dir)


def requirement_digest(req: ExperimentRequirement) -> str:
    raw = json.dumps(req.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
