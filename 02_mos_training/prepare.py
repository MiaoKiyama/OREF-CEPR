"""Build MOS caches with the original cache/reservoir implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_original/mos'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    from tcmmsline3.ms75_rebuild_data import prepare_data, prepare_view_cache
    from tcmmsline3.ms75_sar_joint_data import prepare_joint_support
    from tcmmsline3.convnext_sar_full120_data import prepare_full_support
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Cache output must be new or empty')
    raw = prepare_data(data_root=args.data_root, cache_root=out / 'raw',
                       workers=args.workers, point_capacity=4096, strict_protocol=True)
    views = prepare_view_cache(raw, out / 'views', workers=args.workers, size=512)
    prior = prepare_joint_support(raw, out / 'boundary96', workers=args.workers, capacity=4096)
    boundary = prepare_full_support(raw, prior, out / 'boundary120', workers=args.workers)
    data = dict(data_root=str(Path(args.data_root).resolve()), raw_manifest=str(raw),
                views_cache=str(views), boundary_support=str(boundary))
    (out / 'prepared_data.json').write_text(json.dumps(data, indent=2) + '\n')
    print(out / 'prepared_data.json')


if __name__ == '__main__':
    main()
