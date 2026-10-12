"""Model-assisted source reading with mechanically checked sentence references.

Labels are GENERATED interpretations. Valid references are not proof of
entailment; public human-labelled benchmarks assess that separately.
"""
from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Mapping, Sequence


def reading_sources(works: Sequence[Mapping[str, Any]],
                    documents: Mapping[str, Any], *, max_characters: int = 48000) -> list[dict[str, Any]]:
    """Bound the actual reading input and retain its source offsets.

    Prefer already fetched, identity-bound full text. Do not fetch again or
    turn source authenticity into claim validity. Sentence indices refer to
    this saved excerpt, not to the original paper's sentence numbering.
    """
    from ..livefeed import FreshWork

    unique = {}
    for raw in works:
        work = FreshWork.from_dict(dict(raw))
        sid = work.cite_id()
        if sid and sid not in unique:
            unique[sid] = work
    selected = list(unique.items())[:8]
    if not selected or max_characters <= 0:
        return []
    allowance = min(12000, max_characters // len(selected))
    result = []
    for sid, work in selected:
        document = documents.get(sid) or {}
        bound = (isinstance(document, Mapping) and bool(work.arxiv_id)
                 and document.get('arxiv_id') == work.arxiv_id)
        spans = [(index, span) for index, span in enumerate(document.get('full_text_spans') or [])
                 if isinstance(span, Mapping) and span.get('verified') is True
                 and isinstance(span.get('text'), str) and span['text'].strip()
                 and span.get('source_digest') and span.get('url')] if bound else []
        if spans:
            inputs = [{'field': f'full_text:{index}', **dict(span)} for index, span in spans]
            coverage = 'full_text_excerpt'
        else:
            inputs = [{'field': 'abstract', 'text': work.abstract,
                       'url': work.href(), 'start': 0}]
            coverage = 'abstract_only'
        chunks, provenance, used = [], [], 0
        for span in inputs:
            available = allowance - used - (1 if chunks else 0)
            if available <= 0:
                break
            excerpt = str(span.get('text') or '')[:available]
            if not excerpt.strip():
                continue
            offset = used + (1 if chunks else 0)
            provenance.append({'field': span['field'], 'url': span.get('url', ''),
                'source_digest': span.get('source_digest', ''),
                'source_start': span.get('start', 0), 'characters': len(excerpt),
                'input_start': offset, 'input_end': offset + len(excerpt),
                'truncated': bool(span.get('truncated') or len(excerpt) < len(str(span.get('text') or '')))})
            chunks.append(excerpt)
            used = offset + len(excerpt)
        content = '\n'.join(chunks)
        if not content.strip():
            continue
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
        result.append({'source_id': sid, 'title': work.title,
            'published': str(document.get('published') or work.published) if bound else work.published,
            'text': content, 'sentences': sentences, 'source_spans': provenance,
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'coverage': coverage, 'truncated': any(p['truncated'] for p in provenance)
                or len(provenance) < len(inputs),
            'sentence_index_scope': 'saved_reading_excerpt',
            'epistemic': 'SOURCE_DOCUMENT_NOT_EXPERIMENT'})
    return result


SOURCE_CHECK_SYSTEM = (
    "Read the supplied scientific sources and assess the exact claim. Use only supplied text. "
    "For each source determine SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO. Compare population, "
    "intervention, outcome, effect direction and conditions; distinguish association from "
    "causation. Look for counterevidence and alternative explanations before deciding. "
    "Missing evidence, an unfamiliar idea and a source title are not refutations. "
    "Return JSON {assessments:[{source_id, label, sentences:[zero-based sentence indices], "
    "reason, assumptions:[], unresolved:[]}], remaining_questions:[]}. Select only the "
    "minimal complete evidence sentences; for NOT_ENOUGH_INFO use an empty list. "
    "These are interpretations of sources, not experimental observations or final verdicts."
)


def check_sources(client: Any, claim: str, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from ..llm import complete_call
    from .ideation import parse_json_object
    from .inquiry import strings, text

    clean = []
    for source in sources[:8]:
        sentences = source.get("sentences")
        if (not isinstance(source.get("source_id"), str) or not source["source_id"]
                or not isinstance(sentences, list) or not sentences
                or any(not isinstance(s, str) for s in sentences)):
            raise ValueError("source_check_requires_identified_sentence_lists")
        clean.append({"source_id": source["source_id"], "title": text(source.get("title")),
                      "sentences": list(sentences)})
    by_id = {s["source_id"]: s for s in clean}
    if not clean or len(by_id) != len(clean):
        raise ValueError("source_check_empty_or_duplicate_sources")
    reply = complete_call(client, json.dumps({"claim": claim, "sources": clean}, ensure_ascii=False),
                          purpose="research_source_check", system=SOURCE_CHECK_SYSTEM)
    raw = parse_json_object(reply.text)
    items = raw.get("assessments")
    if not isinstance(items, list):
        raise ValueError("source_assessments_missing")
    rows, seen = [], set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("source_assessment_not_an_object")
        sid, label, indices = item.get("source_id"), item.get("label"), item.get("sentences")
        if not isinstance(sid, str) or sid not in by_id or sid in seen:
            raise ValueError("source_assessment_unknown_or_duplicate_id")
        if label not in {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"}:
            raise ValueError("source_assessment_invalid_label")
        if (not isinstance(indices, list) or len(set(map(str, indices))) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(by_id[sid]["sentences"]) for i in indices)
                or label != "NOT_ENOUGH_INFO" and not indices
                or label == "NOT_ENOUGH_INFO" and indices):
            raise ValueError("source_assessment_invalid_sentence_reference")
        seen.add(sid)
        rows.append({"source_id": sid, "label": label, "sentences": indices,
                     "passages": [by_id[sid]["sentences"][i] for i in indices],
                     "reason": text(item.get("reason")), "assumptions": strings(item.get("assumptions")),
                     "unresolved": strings(item.get("unresolved")), "epistemic": "GENERATED"})
    if seen != set(by_id):
        raise ValueError("source_assessment_incomplete")
    return {"version": "source-check-v1", "claim": claim, "assessments": rows,
            "remaining_questions": strings(raw.get("remaining_questions")),
            "model": str(getattr(reply, "model", "")),
            "completion_digest": str(getattr(reply, "digest", "")),
            "reference_check": "sentence_identity_only", "epistemic": "GENERATED"}
