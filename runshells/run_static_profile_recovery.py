#!/usr/bin/env python3
"""Evaluate Static Profile Recovery on finished checkpoint archives."""
import argparse
import json
import sys
import math
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model.static_profile_recovery import build_pool, prepare, evaluate, read_json, fingerprint, summarize, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--agents-dir', type=Path, default=ROOT / 'frontend/static/assets/village/agents')
    parser.add_argument('--config', type=Path, default=ROOT / 'data/config.json')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=3, help='Attempts per sample per invocation, including first call')
    parser.add_argument('--retry-delay', type=float, default=2, help='Exponential backoff seconds, capped at 60')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='With resume, retry cached errors')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['both', 'all-session', 'single-session'], default='both')
    parser.add_argument('--profile-map', type=Path, help='JSON: run directory name -> patient name -> {persona, severity}')
    parser.add_argument('--summary-output', type=Path, help='Optional cross-run grouped summary JSON')
    parser.add_argument('--dry-run', action='store_true', help='Validate all inputs without model calls or writes')
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error('invalid workers/attempts/retry-delay')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    pool = build_pool(args.agents_dir)
    mapping = read_json(args.profile_map) if args.profile_map else {}
    modes = ('all-session', 'single-session') if args.mode == 'both' else (args.mode,)
    jobs = [(run, prepare(run, pool, args.seed, modes, mapping.get(run.name))) for run in args.run_dirs]
    if args.dry_run:
        print(json.dumps({'profiles': len(pool['profiles']), 'pool_sizes': {k: len(v) for k, v in pool['pools'].items()},
                          'pool_audit': pool['audit'], 'runs': [{'run': str(r), 'samples': len(s)} for r, s in jobs]}, ensure_ascii=False, indent=2))
        return 0
    for run, _ in jobs:
        if (run / 'humanlike/static_profile_recovery').exists() and not args.resume:
            parser.error(f'Output already exists: {run}/humanlike/static_profile_recovery')
    if args.summary_output and args.summary_output.exists() and not args.resume:
        parser.error(f'Summary already exists: {args.summary_output}')
    from modules.model.llm_model import create_llm_model, sanitize_endpoint_for_log
    config = read_json(args.config)['agent']['think']['llm']
    from modules.model.endpoint_pool import resolve_endpoint_urls
    endpoints = [sanitize_endpoint_for_log(url) for url in resolve_endpoint_urls(config)]
    print('Judge endpoints:', endpoints, 'workers:', args.workers)
    local = threading.local()
    metadata = {'model': config['model'], 'base_url': sanitize_endpoint_for_log(config['base_url']),
                'endpoints': endpoints, 'config_hash': fingerprint(config), 'temperature': 0, 'config_source': str(args.config.resolve())}
    def call(prompt, field):
        if not hasattr(local, 'model'):
            local.model = create_llm_model(config)
        return local.model.completion(prompt, caller=f'static_profile_recovery_{field}', temperature=0, retry=1, failsafe=None)
    failed = False
    for run, samples in jobs:
        result = evaluate(run, pool, samples, call, metadata, workers=args.workers,
                          attempts=args.attempts, retry_delay=args.retry_delay,
                          resume=args.resume, retry_errors=args.retry_errors)
        print(json.dumps({'run': str(run), **result}, ensure_ascii=False))
        failed |= result['status'] != 'complete'
    if args.summary_output:
        rows = [json.loads(line) for run, _ in jobs
                for line in (run / 'humanlike/static_profile_recovery/samples.jsonl').read_text(encoding='utf-8').splitlines()]
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.summary_output, {'overall': summarize(rows),
                    **{k: summarize(rows, (k,)) for k in ('persona', 'severity', 'group')},
                    'joint': summarize(rows, ('persona', 'severity', 'group'))})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
