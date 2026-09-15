#!/usr/bin/env python3
"""Evaluate finished checkpoint archives using the current local think-llm."""
import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model.psi_bench_eval import load_sessions, read_json, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path, help='Finished checkpoint run directories')
    parser.add_argument('--config', type=Path, default=ROOT / 'data/config.json')
    parser.add_argument('--ptc-history', type=int, default=6)
    parser.add_argument('--emotion-history', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3, help='Concurrent independent judge calls')
    parser.add_argument('--attempts', type=int, default=3, help='Total attempts per classification per invocation')
    parser.add_argument('--retry-delay', type=float, default=2, help='Exponential backoff base seconds, capped at 60')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='With resume, retry previously failed classifications')
    parser.add_argument('--dry-run', action='store_true', help='Validate extraction without model calls or writes')
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or args.retry_delay < 0:
        parser.error('workers/attempts must be positive and retry-delay nonnegative')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if min(args.ptc_history, args.emotion_history) < 0:
        parser.error('history limits must be nonnegative')
    if args.dry_run:
        for path in args.run_dirs:
            sessions, audit = load_sessions(path)
            print(json.dumps({'run': str(path), 'sessions': len(sessions),
                              'patient_turns': sum(m['role'] == 'patient' for s in sessions for m in s['messages']),
                              'audit': audit}, ensure_ascii=False, indent=2))
        return 0
    from modules.model.llm_model import create_llm_model, sanitize_endpoint_for_log
    config = read_json(args.config)['agent']['think']['llm']
    from modules.model.endpoint_pool import resolve_endpoint_urls
    local = threading.local()
    print('Judge endpoints:', [sanitize_endpoint_for_log(url) for url in resolve_endpoint_urls(config)])
    def call(prompt, kind):
        # LLMModel has mutable per-call state: never share an instance across threads.
        if not hasattr(local, 'model'):
            local.model = create_llm_model(config)
        return local.model.completion(prompt, caller=f'psi_bench_{kind}', temperature=0,
                                      retry=1, failsafe=None)
    failed = False
    for path in args.run_dirs:
        result = run_evaluation(path, call, ptc_history=args.ptc_history,
                                emotion_history=args.emotion_history, workers=args.workers,
                                attempts=args.attempts, retry_delay=args.retry_delay,
                                resume=args.resume, retry_errors=args.retry_errors,
                                metadata={'config': str(args.config.resolve()), 'model': config['model'],
                                          'base_url': sanitize_endpoint_for_log(config['base_url']),
                                          'temperature': 0,
                                          'endpoints': [sanitize_endpoint_for_log(url) for url in resolve_endpoint_urls(config)],
                                          'thinking': config.get('thinking'),
                                          'reasoning_effort': config.get('reasoning_effort'),
                                          'caller_overrides': config.get('caller_overrides', {})})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failed |= result['status'] != 'complete'
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
