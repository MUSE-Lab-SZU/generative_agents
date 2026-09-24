#!/usr/bin/env python3
"""Evaluate Static Profile Recovery on finished checkpoint archives."""
import argparse
import json
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model.static_profile_recovery import build_pool, prepare, evaluate, read_json, summarize, atomic_json
from runshells.run_psi_bench_eval import (
    add_judge_arguments, add_run_arguments, select_run_dirs,
    load_judge_settings, make_judge_call,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    add_judge_arguments(parser)
    parser.add_argument('--agents-dir', type=Path, default=ROOT / 'frontend/static/assets/village/agents')
    parser.add_argument('--source', choices=['consult', 'resident'], default='consult')
    parser.add_argument('--chat-kind', choices=['all', 'forced', 'spontaneous'], default='all',
                        help='Resident source only; unknown event type is included only with all')
    parser.add_argument('--patient-name', help='Target name in conversation.json; default from archived manifest or 卡布达')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=3, help='Attempts per sample per invocation, including first call')
    parser.add_argument('--retry-delay', type=float, default=2, help='Exponential backoff seconds, capped at 60')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='With resume, retry cached errors')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['both', 'all-session', 'single-session'], default='single-session')
    parser.add_argument('--profile-map', type=Path, help='JSON: run directory name -> patient name -> {persona, severity}')
    parser.add_argument('--summary-output', type=Path, help='Optional cross-run grouped summary JSON')
    parser.add_argument('--dry-run', action='store_true', help='Validate all inputs without model calls or writes')
    parser.add_argument('--output-root', type=Path, help='Store all run outputs under this separate directory')
    parser.add_argument('--show-prompts', type=int, default=0, metavar='N',
                        help='Print first N actual recovery prompts; no model calls or writes')
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error('invalid workers/attempts/retry-delay')
    if args.show_prompts < 0:
        parser.error('--show-prompts must be nonnegative')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if args.source == 'consult' and (args.chat_kind != 'all' or args.patient_name):
        parser.error('--chat-kind and --patient-name require --source resident')
    try:
        runs = select_run_dirs(args.run_dirs, args.run_date, args.checkpoints_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Selected {len(runs)} checkpoint run(s):', *[path.name for path in runs], sep='\n  ')
    selected_backend = args.backend
    output_subdir = (f'static_profile_recovery_v3_consult_{selected_backend}' if args.source == 'consult' else
                     f'static_profile_recovery_v3_resident_{args.chat_kind}_{selected_backend}')
    pool = build_pool(args.agents_dir)
    mapping = read_json(args.profile_map) if args.profile_map else {}
    modes = ('all-session', 'single-session') if args.mode == 'both' else (args.mode,)
    jobs = [(run, prepare(run, pool, args.seed, modes, mapping.get(run.name),
                          source=args.source, chat_kind=args.chat_kind, patient_name=args.patient_name))
            for run in runs]
    if args.dry_run or args.show_prompts:
        for run, samples in jobs:
            print(json.dumps({'run': str(run), 'source': args.source, 'chat_kind': args.chat_kind,
                              'profiles': len(pool['profiles']), 'samples': len(samples),
                              'by_interaction_type': {kind: sum(s['interaction_type'] == kind for s in samples)
                                                      for kind in sorted({s['interaction_type'] for s in samples})},
                              'pool_audit': pool['audit']}, ensure_ascii=False, indent=2))
            for sample in samples[:args.show_prompts]:
                print(json.dumps({'run': str(run), 'sample_id': sample['sample_id'],
                                  'field': sample['field'], 'mode': sample['mode'],
                                  'source_type': sample['source_type'],
                                  'interaction_type': sample['interaction_type'],
                                  'prompt': sample['prompt']}, ensure_ascii=False, indent=2))
        return 0
    for run, _ in jobs:
        output = (args.output_root / run.name / output_subdir if args.output_root else
                  run / 'humanlike' / output_subdir)
        if output.exists() and not args.resume:
            parser.error(f'Output already exists: {output}')
    if args.summary_output and args.summary_output.exists() and not args.resume:
        parser.error(f'Summary already exists: {args.summary_output}')
    try:
        routing, key, metadata = load_judge_settings(args)
        call = make_judge_call(routing, key, metadata)
    except ValueError as exc:
        parser.error(str(exc))
    print('Judge endpoints:', metadata['endpoints'], 'workers:', args.workers)
    failed = False
    for run, samples in jobs:
        result = evaluate(run, pool, samples, call, metadata, workers=args.workers,
                          attempts=args.attempts, retry_delay=args.retry_delay,
                          resume=args.resume, retry_errors=args.retry_errors,
                          output_subdir=output_subdir, output_root=args.output_root)
        print(json.dumps({'run': str(run), **result}, ensure_ascii=False))
        failed |= result['status'] != 'complete'
    if args.summary_output:
        rows = [json.loads(line) for run, _ in jobs
                for line in ((args.output_root / run.name / output_subdir if args.output_root else
                              run / 'humanlike' / output_subdir) / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.summary_output, {'overall': summarize(rows),
                    **{k: summarize(rows, (k,)) for k in ('persona', 'severity', 'group')},
                    'joint': summarize(rows, ('persona', 'severity', 'group')),
                    'interaction_types': summarize(rows, ('source_type', 'interaction_type'))})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
