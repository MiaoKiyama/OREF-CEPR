"""Launch the released trainers with verified source bindings.

Sampling, losses, optimizer, schedules, EMA and checkpoint mathematics remain
unchanged. Each process imports one isolated project.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def verify_sources(project):
    records = json.loads((ROOT.parent / 'docs/original_training_hashes.json').read_text())
    selected = {k: v for k, v in records.items() if k.startswith('training_original/' + project + '/')}
    for name, entry in selected.items():
        if hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Bound training source changed: ' + name)
    return {k: v['sha256'] for k, v in selected.items()}


def install_dino_release_binding(module, sources):
    # Archived held24/source-manifest paths are provenance, not training inputs.
    # Authenticate the published source snapshot instead; retain data and weight
    # audits, scientific config validation and strict recovery binding.
    from tcmmsline3 import dino_web_protocol as protocol
    from tcmmsline3.runtime import config_fingerprint
    import torch

    def load(path):
        config = protocol.load_dino_web_config(path, require_pretrained_file=False)
        model = config['model']
        if protocol.sha256_file(model['pretrained_file']) != model['pretrained_file_sha256']:
            raise ValueError('Pretrained checkpoint digest mismatch')
        audit = module.audit_dino_web_data_identity(config)
        binding = dict(role='released_original_training', sources=sources,
                       config_fingerprint=config_fingerprint(config), data_identity=audit,
                       pretrained_file_sha256=model['pretrained_file_sha256'])
        output = Path(config['experiment']['output_dir'])
        previous = output / 'runtime_binding.json'
        if previous.exists() and json.loads(previous.read_text()) != binding:
            raise RuntimeError('Output belongs to a different configuration')
        module._CONFIG, module._RUNTIME_BINDING = config, binding
        module.atomic_json_dump(binding, previous)
        module.atomic_json_dump(audit, output / 'data_identity.json')
        torch.set_num_threads(config.get('pipeline', {}).get('torch_threads', 2))
        torch.set_num_interop_threads(1)
        return config

    module.load_bound_dino_web_config = load


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('project', choices=['dino', 'mos', 'fivefold', 'oref'])
    parser.add_argument('--expert', choices=['dino', 'mos'], default='dino')
    args, remaining = parser.parse_known_args()
    sources = verify_sources(args.project)
    project = ROOT / args.project
    sys.path.insert(0, str(project))
    if args.project == 'dino':
        module = importlib.import_module('tools.train_dino_web')
        install_dino_release_binding(module, sources)
    elif args.project == 'mos':
        module = importlib.import_module('tools.native24_retrain_limited')
        if '--config' in remaining:
            config = json.loads(Path(remaining[remaining.index('--config') + 1]).read_text())
            indices, uuids = config['physical_gpus'], config['gpu_uuids']
            if len(indices) != 2 or len(set(indices)) != 2 or len(uuids) != 2:
                raise ValueError('Original MOS training requires two distinct GPUs')
            module.JOBS['pyramid']['gpus'] = indices
            module.GPU_UUID_BY_INDEX = dict(zip(indices, uuids))
    elif args.project == 'fivefold':
        module = importlib.import_module('tools.train_native24_oof_' + ('dino' if args.expert == 'dino' else 'pyramid'))
        if args.expert == 'mos' and '--config' in remaining:
            config = json.loads(Path(remaining[remaining.index('--config') + 1]).read_text())
            if len(config['physical_gpus']) != 1 or len(config['gpu_uuids']) != 1:
                raise ValueError('Original OOF MOS training requires one GPU')
            module.AUTHORIZED_GPUS = tuple(config['physical_gpus'])
            module.GPU_UUID_BY_INDEX = {config['physical_gpus'][0]: config['gpu_uuids'][0]}
    else:
        module = importlib.import_module('native24_object_graph_worker')
    sys.argv = [module.__file__, *remaining]
    module.main()


if __name__ == '__main__':
    main()
