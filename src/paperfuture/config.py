"""Explicit run configuration: no inferred model, cutoff or domain."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from .storage import read_json, timestamp


DOMAINS = {
    "agent_security": {
        "label": "Agent security",
        "queries": ['all:"prompt injection"', 'all:"agent security"', 'all:"tool use" AND all:"security"'],
        "facets": ["threat model", "instruction/data boundary", "tool permissions", "adaptive attack", "defense evaluation"],
    },
    "multimodal_self_evolution_xai": {
        "label": "多模态大模型自进化中的可解释性",
        "queries": ['all:"multimodal" AND all:"self improvement"', 'all:"vision language" AND all:"interpretability"',
                    'all:"multimodal" AND all:"self evolution"'],
        "facets": ["self improvement mechanism", "cross-modal attribution", "feedback reliability", "representation change", "explanation faithfulness"],
    },
    "ai4eda": {
        "label": "AI for electronic design automation",
        "queries": ['all:"large language model" AND all:"Verilog"', 'all:"machine learning" AND all:"electronic design automation"',
                    'all:"RTL" AND all:"verification"'],
        "facets": ["RTL generation", "functional verification", "synthesis constraints", "PPA optimization", "design generalization"],
    },
}
ROLES = ("extract", "rerank", "graph", "graph_stages", "graph_merge", "graph_edges", "graph_review", "propose",
         "simulate", "simulate_secondary", "critique", "critique_secondary", "discuss", "assess")
DEFAULTS = {
    "mode": "prospective", "max_rounds": 4, "replicate_id": "1", "api_mode": "live",
    "retrieval": {"method": "hybrid", "recall_k": 160, "candidate_limit": 100, "selected_limit": 40,
                  "min_relevance": 2, "min_quality": 2, "rrf_k": 60, "rerank_batch": 12,
                  "coverage_weight": 0.3, "redundancy_weight": 0.2, "recent_years": 2,
                  "recent_fraction": 0.65, "extraction_chars": 22000, "round_papers": 8,
                  "round_evidence_chars": 26000, "hydrate_candidates": 30},
    "graph": {"stage_batch": 8, "predecessors": 4, "edge_batch": 1, "review_batch": 1, "concurrent_calls": 3},
    "world_model": {"simulators": ["simulate", "simulate_secondary"], "critics": ["critique"],
                    "discussion": True, "min_distinct_models": 2},
    "policy": {"critic_weight": 0.2, "sufficient_utility": 0.75, "min_novelty": 3,
               "min_importance": 3, "hard_conflict_severity": 0.8, "stagnation_epsilon": 0.02},
    "api": {"max_calls": 500, "max_tokens": 2000000, "max_output_tokens": 6000,
            "timeout_seconds": 180, "json_attempts": 2, "max_context_chars": 180000},
    "sources": [], "models": {}, "model_vintage_evidence": [],
}


def _expand(value):
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError(f"configuration requires environment variable {name}")
            return os.environ[name]
        return re.sub(r"\$\{([A-Z_][A-Z_0-9]*)\}", replace, value)
    return value


def load_config(path: Path) -> dict:
    config = normalize(_expand(read_json(path)))
    base = path.resolve().parent
    for source in config["sources"]:
        if source["type"] == "jsonl":
            source["path"] = str((base / source["path"]).resolve())
    for model in config["models"].values():
        if model.get("api_key_file"):
            model["api_key_file"] = str((base / Path(model["api_key_file"]).expanduser()).resolve())
    return config


def normalize(raw: dict) -> dict:
    config = copy.deepcopy(DEFAULTS)
    allowed = set(DEFAULTS) | {"run_id", "domain", "question", "cutoff", "subfield"}
    if set(raw) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(raw)-allowed)}")
    for key, value in raw.items():
        if key in ("retrieval", "policy", "api", "graph", "world_model"):
            if set(value) - set(config[key]):
                raise ValueError(f"unknown {key} configuration")
            config[key].update(value)
        else:
            config[key] = value
    for key in ("run_id", "question", "cutoff", "domain"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"required configuration: {key}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", config["run_id"]):
        raise ValueError("run_id must be a safe filename of at most 96 characters")
    if config["domain"] not in DOMAINS:
        raise ValueError(f"domain must be one of {list(DOMAINS)}")
    timestamp(config["cutoff"])
    if type(config["max_rounds"]) is not int or not 1 <= config["max_rounds"] <= 4:
        raise ValueError("max_rounds is a GLOBAL integer limit from 1 through 4")
    if config["mode"] not in ("prospective", "retrospective") or config["api_mode"] not in ("live", "replay"):
        raise ValueError("invalid run mode")
    if config["retrieval"]["method"] not in ("hybrid", "lexical"):
        raise ValueError("retrieval method must be hybrid or explicit lexical ablation")
    if not 0 <= config["policy"]["critic_weight"] <= 0.35:
        raise ValueError("critic_weight must be within [0, 0.35]; keep exploration value explicit")
    for key in ("sufficient_utility", "hard_conflict_severity", "stagnation_epsilon"):
        if not 0 <= config["policy"][key] <= 1:
            raise ValueError(f"policy.{key} must be within [0, 1]")
    for key in ("min_novelty", "min_importance"):
        if type(config["policy"][key]) is not int or not 0 <= config["policy"][key] <= 4:
            raise ValueError(f"policy.{key} must be an integer within [0, 4]")
    for key in ("recall_k", "candidate_limit", "selected_limit", "rrf_k", "rerank_batch"):
        if type(config["retrieval"][key]) is not int or config["retrieval"][key] < 1:
            raise ValueError(f"retrieval.{key} must be a positive integer")
    for key in ("recent_years", "extraction_chars", "round_papers", "round_evidence_chars"):
        if type(config["retrieval"][key]) is not int or config["retrieval"][key] < 1:
            raise ValueError(f"retrieval.{key} must be positive")
    if type(config["retrieval"]["hydrate_candidates"]) is not int or config["retrieval"]["hydrate_candidates"] < 0:
        raise ValueError("hydrate_candidates must be a nonnegative integer")
    if not 0 <= config["retrieval"]["recent_fraction"] <= 1:
        raise ValueError("recent_fraction must be in [0, 1]")
    for key, value in config["graph"].items():
        if type(value) is not int or value < 1:
            raise ValueError(f"graph.{key} must be positive")
    if config["graph"]["concurrent_calls"] > 8:
        raise ValueError("graph.concurrent_calls cannot exceed 8")
    for key, value in config["api"].items():
        if type(value) is not int or value < 1:
            raise ValueError(f"api.{key} must be a positive integer")
    if config["api"]["json_attempts"] > 3:
        raise ValueError("json_attempts cannot exceed 3")
    if config["api"]["max_tokens"] < config["api"]["max_output_tokens"]:
        raise ValueError("max_tokens must accommodate at least one output allowance")
    for source in config["sources"]:
        if source.get("type") == "jsonl":
            if set(source) != {"type", "path"}:
                raise ValueError("jsonl source expects only type and path")
        elif source.get("type") == "arxiv_dataset":
            if set(source)-{"type","data_root","historical_limit","recent_limit","fulltext_limit"} or not source.get("data_root"):
                raise ValueError("arxiv_dataset requires a data_root and optional acquisition limits")
        elif source.get("type") == "arxiv":
            if set(source) - {"type", "queries", "ids", "max_results", "page_size", "start_date", "version_policy", "full_text", "require_full_text"}:
                raise ValueError("unknown arxiv source option")
            if source.get("version_policy", "v1") != "v1":
                raise ValueError("automatic arxiv collection currently freezes v1; import audited later versions as JSONL")
            if source.get("full_text", "none") not in ("none", "pdf"):
                raise ValueError("arxiv full_text must be none or pdf (requires pdftotext)")
            if source.get("require_full_text", False) and source.get("full_text") != "pdf":
                raise ValueError("require_full_text needs full_text=pdf")
        else:
            raise ValueError("sources must be jsonl or arxiv")
    if not config["sources"]:
        raise ValueError("at least one literature source is required")
    models = config["models"]
    if "default" not in models:
        raise ValueError("models.default must specify a base_url, model and credential reference")
    if set(models) - set(ROLES) - {"default", "embedding"}:
        raise ValueError("unknown model role")
    if config["retrieval"]["method"] == "hybrid" and "embedding" not in models:
        raise ValueError("hybrid retrieval needs models.embedding; choose lexical explicitly for an ablation")
    for role, model in models.items():
        if set(model) - {"base_url", "model", "api_key_env", "api_key_file"}:
            raise ValueError(f"models.{role}: unsupported option (never put a plaintext key in config)")
        if not model.get("model") or not model.get("base_url"):
            raise ValueError(f"models.{role}: model and base_url are required")
        url = urlparse(model["base_url"])
        if url.scheme not in ("https", "http") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) endpoint without embedded credentials or query")
        model["base_url"] = model["base_url"].rstrip("/")
        if not model.get("api_key_env") and not model.get("api_key_file"):
            raise ValueError(f"models.{role}: api_key_env or api_key_file is required")
    panel = config["world_model"]
    for key in ("simulators", "critics"):
        values = panel[key]
        if not isinstance(values, list) or not values or len(values) > 3 or len(set(values)) != len(values):
            raise ValueError(f"world_model.{key} must name one to three distinct model roles")
        if any(role not in ROLES for role in values):
            raise ValueError("unknown world-model role")
    if type(panel["discussion"]) is not bool or type(panel["min_distinct_models"]) is not int or panel["min_distinct_models"] < 1:
        raise ValueError("invalid world-model discussion/diversity configuration")
    identities = {(models.get(role, models["default"])["base_url"], models.get(role, models["default"])["model"])
                  for role in panel["simulators"] + panel["critics"]}
    if len(identities) < panel["min_distinct_models"]:
        raise ValueError("world-model panel requires distinct configured model IDs; repeated roles are not distinct models")
    return config


def credential(model: dict) -> str:
    value = os.environ.get(model.get("api_key_env", ""), "").strip()
    if not value and model.get("api_key_file"):
        value = Path(model["api_key_file"]).expanduser().read_text().strip()
    if not value:
        raise ValueError("API credential unavailable; configure the referenced environment variable or key file")
    return value


def public_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    # Locations and environment names are allowed; secrets are never serialized.
    result.pop("api_mode", None)  # live -> replay does not change the scientific run.
    return result
