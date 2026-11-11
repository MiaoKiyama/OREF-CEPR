"""Relocate an archived configuration without changing its training recipe.

Pass explicit OLD=NEW path prefixes. Existing binary/data bindings must still
match their recorded SHA256. Source bindings are rebuilt for the byte-verified
published source project. Cache preparation is separate; this does not invent
new crop indices, boundary reservoirs, epoch plans or split memberships.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from launch import ROOT, verify_sources


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, choices=['dino', 'mos', 'fivefold'])
    p.add_argument('--template', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--run-dir', required=True, help='New, empty experiment directory')
    p.add_argument('--map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--old-project', help='Archived source checkout prefix to replace')
    p.add_argument('--prepared-data', help='MOS prepared_data.json from the original cache builders')
    args = p.parse_args()
    mappings = [tuple(x.split('=', 1)) for x in args.map]
    if any(len(x) != 2 or not x[0] or not x[1] for x in mappings):
        p.error('--map requires nonempty OLD=NEW')
    if args.old_project:
        mappings.append((args.old_project, str(ROOT / args.project)))
    original = json.loads(Path(args.template).read_text())
    sources = verify_sources(args.project)

    def replace(value):
        if isinstance(value, str):
            for old, new in sorted(mappings, key=lambda x: -len(x[0])):
                if value == old or value.startswith(old.rstrip('/') + '/'):
                    return new.rstrip('/') + value[len(old.rstrip('/')):]
            return value
        if isinstance(value, list):
            return [replace(x) for x in value]
        if isinstance(value, dict):
            return {replace(k): replace(v) for k, v in value.items()}
        return value

    config = replace(copy.deepcopy(original))
    prepared = {}
    if args.prepared_data:
        if args.project != 'mos':
            p.error('--prepared-data is only for full120 MOS')
        prepared = json.loads(Path(args.prepared_data).read_text())
        if set(prepared) != {'data_root', 'raw_manifest', 'views_cache', 'boundary_support'}:
            raise ValueError('Unexpected cache configuration fields')
        config['data'].update(prepared)
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Run directory must be new or empty')
    if args.project == 'dino':
        config['experiment']['output_dir'] = str(run)
        config['train']['resume_checkpoint'] = None
    else:
        if args.project == 'mos' and ('native24_retrain_20261015_v1' not in run.parts or run.name != 'pyramid'):
            raise ValueError('MOS run-dir must end with native24_retrain_20261015_v1/pyramid')
        config['output_dir'] = str(run)
        config['checkpoint_dir'] = str(run if args.project == 'mos' else run / 'recovery')
        config['candidate_dir'] = str(run / 'candidates')
        if 'storage_roots' in config:
            config['storage_roots'] = [str(run / config['campaign_name'] / 'payloads')]
        if 'checkpoint_lock_file' in config:
            config['checkpoint_lock_file'] = str(run / 'checkpoint.lock')
    # Source identity changes with the checkout layout, but source bytes do not.
    for key in ('files', 'source_files'):
        if key not in config:
            continue
        bound = {}
        for old_path, digest in original[key].items():
            if args.old_project and old_path.startswith(args.old_project.rstrip('/') + '/'):
                continue
            if prepared and old_path in [original['data'][k] for k in ('raw_manifest', 'views_cache', 'boundary_support')]:
                continue
            path = replace(old_path)
            if sha(path) != digest:
                raise ValueError('Bound input differs: ' + path)
            bound[path] = digest
        bound.update({str(ROOT.parent / k): v for k, v in sources.items()})
        if prepared:
            bound.update({prepared[k]: sha(prepared[k]) for k in ('raw_manifest', 'views_cache', 'boundary_support')})
        config[key] = bound
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A separate new config is required; never overwrite an experiment.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')
    print(output)


if __name__ == '__main__':
    main()
