"""Verify the copied snapshot and run one offline remote-search smoke check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    mismatches = []
    for row in manifest['files']:
        path = ROOT/row['path']
        if (path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
            mismatches.append(row['path'])
    from farfield.extras.openworld.remote import ConceptSpaceSearch
    result = ConceptSpaceSearch('attn-concepts-s1', root=ROOT)(
        purpose='snapshot_smoke', intent='attention mechanism and long context', k=3).to_dict()
    external = [name for name, module in list(sys.modules.items())
                if (name == 'farfield' or name.startswith('farfield.'))
                and getattr(module, '__file__', None)
                and not Path(module.__file__).resolve().is_relative_to(ROOT/'src')]
    report = {'copied_files':len(manifest['files']), 'hash_mismatches':mismatches,
              'external_farfield_imports':external, 'remote_status':result['status'],
              'corpus':result['corpus'], 'frames_returned':len(result['frames']),
              'model_calls':0, 'scope':'copy_integrity_and_offline_import_smoke_not_scientific_evaluation'}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(report,stream,indent=2)
            stream.write('\n')
    print(json.dumps(report,ensure_ascii=False))
    return 1 if mismatches or external or not result['frames'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
