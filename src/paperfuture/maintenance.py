"""Scheduled corpus maintenance, separate from every research run.

Only arXiv acquisition and loopback embedding calls are allowed here. Default
profiles are advanced after validation and warming; previous profiles and all
run artifacts remain available for exact resumption.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from .api import ModelGateway
from .config import load_config, normalize
from .dataset import load_corpus, update_corpus
from .storage import atomic_json, digest, exclusive, now, read_json, seal, timestamp


def pending_ids(root, domain, known, configs, limit):
    candidates = []
    for config in configs:
        candidates.extend(config['retrieval']['history_paper_ids'])
    # Newer research queues first; coverage labels are diagnostics, not queries
    # or executable instructions. Only canonical paper IDs can be fetched.
    paths = sorted((root/'runs').glob('*/literature_update_requests.json'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        value = read_json(path)
        if value.get('domain') == domain:
            candidates.extend(value.get('uncollected_history_seeds', []))
            candidates.extend(value.get('uncollected_references', {}))
    accepted = []
    for value in candidates:
        if not isinstance(value, str) or not re.fullmatch(r'arxiv:\d{4}\.\d{4,5}(?:@v1)?', value):
            continue
        value = value.removesuffix('@v1')
        if value not in known and value not in accepted:
            accepted.append(value)
    return accepted[:limit]


def warm_release(root, config, manifest_path):
    """Warm exact document rows using the local model, with resumable receipts."""
    manifest, papers = load_corpus(manifest_path)
    config = copy.deepcopy(config)
    model = config['models']['embedding']
    endpoint = urlparse(model['base_url'])
    if endpoint.hostname != '127.0.0.1' or endpoint.scheme != 'http' or model.get('cache_scope') != 'text':
        raise ValueError('weekly maintenance requires a loopback embedding model with cache_scope=text')
    config['api_mode'] = 'live'
    config['api'].update(max_calls=1000, max_tokens=100_000_000, timeout_seconds=600)
    identity = digest({'release_id': manifest['release_id'], 'model': model, 'question': config['question']})
    directory = root/'runs'/('local-corpus-index-'+manifest['domain']+'-'+identity[:12])
    with exclusive(directory/'.lock'):
        seal(directory/'spec.json', {'artifact_type':'local_embedding_cache_maintenance_not_research_run',
            'release_id':manifest['release_id'], 'models':{'embedding':model}, 'input_hash':identity})
        gateway = ModelGateway(config, directory/'api', identity)
        try:
            for offset in range(0, len(papers), 32):
                gateway.embeddings([p['title']+'\n'+p['sections']['abstract'] for p in papers[offset:offset+32]],
                                   identity=f'documents:{identity}:{offset}')
                atomic_json(directory/'progress.json', {'at':now(), 'total_papers':len(papers),
                    'complete_papers':min(offset+32,len(papers))})
            gateway.embeddings([config['question']], identity=f'query:{identity}', input_type='query')
            result = {'status':'complete', 'papers':len(papers), 'release_id':manifest['release_id'],
                'external_model_calls':0, 'usage':gateway.usage(), 'directory':str(directory)}
            atomic_json(directory/'completed.json', result)
            return result
        finally:
            gateway.close()


def promote_profiles(root, paths, originals, release):
    """Archive old runnable configs, then atomically replace each default."""
    # Do not silently discard a user's edits made while the update was running.
    if any(read_json(path) != original for path, original in zip(paths, originals)):
        raise ValueError('default profile changed during maintenance; activation cancelled')
    promoted = []
    for path, original in zip(paths, originals):
        archive = root/'configs/history'/(path.stem+'.'+digest(original)[:16]+'.json')
        if archive.exists() and read_json(archive) != original:
            raise ValueError('archived profile hash conflict')
        if not archive.exists():
            atomic_json(archive, original)
        updated = copy.deepcopy(original)
        updated.update(literature_mode='frozen', cutoff=release['cutoff'],
                       sources=[{'type':'corpus', 'manifest':release['manifest_path']}])
        updated['run_id'] = (release['domain']+'-corpus-'+release['release_id'][:12]+'-'+
                             digest({k:v for k,v in updated.items() if k != 'run_id'})[:6])
        normalize(updated)
        atomic_json(path, updated)
        promoted.append({'profile':str(path), 'archived_profile':str(archive), 'run_id':updated['run_id']})
    atomic_json(root/'corpora'/release['domain']/'active.json', {
        'release_id':release['release_id'], 'manifest_path':release['manifest_path'],
        'activated_at':now(), 'profiles':promoted})
    return promoted


def maintain(config_path, *, cutoff=None, check_only=False):
    settings = read_json(config_path)
    root = Path(settings['data_root']).resolve()
    cutoff = cutoff or dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0).isoformat().replace('+00:00','Z')
    overlap = settings.get('overlap_days',7)
    fulltexts = settings.get('fulltext_limit',24)
    extra_limit = settings.get('supplement_limit',20)
    if not 0 <= overlap <= 365 or not 0 <= fulltexts <= 1000 or not 0 <= extra_limit <= 100:
        raise ValueError('invalid weekly maintenance limits')
    groups = {}
    for value in settings['profiles']:
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('maintenance profiles must live on the declared data volume')
        if any(path in item['paths'] for item in groups.values()):
            raise ValueError('duplicate maintenance profile')
        config = load_config(path)
        if config['literature_mode'] != 'frozen' or len(config['sources']) != 1 or config['sources'][0]['type'] != 'corpus':
            raise ValueError('weekly defaults must pin one immutable corpus release')
        manifest_path = Path(config['sources'][0]['manifest']).resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError('corpus release must live on the data volume')
        manifest, records = load_corpus(manifest_path)
        if manifest['domain'] != config['domain']:
            raise ValueError('corpus domain mismatch')
        model = config['models']['embedding']
        if urlparse(model['base_url']).hostname != '127.0.0.1' or model.get('cache_scope') != 'text':
            raise ValueError('weekly warming must use a local per-text embedding cache')
        group = groups.setdefault(config['domain'], {'paths':[], 'configs':[], 'originals':[], 'bases':[]})
        group['paths'].append(path); group['configs'].append(config); group['originals'].append(read_json(path))
        group['bases'].append((manifest_path, manifest, records))
    if not groups:
        raise ValueError('no weekly corpus profiles configured')
    report = {'started_at':now(), 'cutoff':cutoff, 'check_only':check_only, 'domains':[],
              'external_model_calls':0, 'schedule':'weekly; timer configuration is separate'}
    if check_only:
        report['domains'] = [{'domain':name, 'profiles':[str(p) for p in g['paths']],
            'papers':[b[1]['papers'] for b in g['bases']]} for name,g in groups.items()]
        report['status'] = 'configuration_validated_without_network'
        return report
    with exclusive(root/'maintenance/.lock'):
        report_path = root/'maintenance/reports'/(cutoff.replace(':','').replace('+','_')+'.json')
        for domain, group in groups.items():
            entry = {'domain':domain, 'status':'running'}
            report['domains'].append(entry)
            try:
                # A prior interrupted activation may have advanced only one
                # alias. Resume from the newest valid release and repair aliases.
                path, base, records = max(group['bases'], key=lambda b: timestamp(b[1]['cutoff']))
                if timestamp(cutoff) < timestamp(base['cutoff']):
                    raise ValueError('maintenance cutoff precedes the active corpus')
                if timestamp(cutoff) == timestamp(base['cutoff']):
                    release = {**base, 'manifest_path':str(path)}
                else:
                    ids = pending_ids(root, domain, {p['paper_id'] for p in records}, group['configs'], extra_limit)
                    release = update_corpus(path, root, cutoff, overlap_days=overlap,
                                            fulltext_limit=fulltexts, extra_ids=ids)
                entry['release_id'] = release['release_id']
                warmed = set()
                entry['warming'] = []
                for config in group['configs']:
                    key = digest({'model':config['models']['embedding'], 'question':config['question']})
                    if key not in warmed:
                        entry['warming'].append(warm_release(root, config, Path(release['manifest_path'])))
                        warmed.add(key)
                entry['profiles'] = promote_profiles(root, group['paths'], group['originals'], release)
                entry.update(status='activated', papers=release['papers'], fulltext_papers=release['fulltext_papers'])
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                # Provider response bodies/credentials are intentionally absent.
                entry.update(status='failed', error_type=type(exc).__name__,
                             failure_policy='existing research snapshots and earlier profiles preserved; inspect acquisition audit/local cache progress')
            atomic_json(report_path, report)
        report.update(completed_at=now(), status='complete' if all(x['status']=='activated' for x in report['domains']) else 'partial_failure')
        atomic_json(report_path, report)
        atomic_json(root/'maintenance/latest.json', report)
    return report
