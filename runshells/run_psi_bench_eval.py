#!/usr/bin/env python3
"""Evaluate finished checkpoint archives using a judge configured by this CLI."""
import argparse
import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model.endpoint_pool import next_endpoint, resolve_endpoint_urls
from modules.model.psi_bench_eval import fingerprint, load_sessions, make_prompt, run_evaluation


def add_judge_arguments(parser):
    parser.add_argument('--backend', choices=['local_vllm', 'deepseek_api'], default='local_vllm')
    parser.add_argument('--local-model', default='qwen3.5-9b-vllm')
    parser.add_argument('--local-base-url', default='http://127.0.0.1:18000/v1')
    parser.add_argument('--local-ports', default='18000', help='Comma-separated local vLLM ports; empty uses base URL')
    parser.add_argument('--local-api-key', default='EMPTY')
    parser.add_argument('--deepseek-model', default='deepseek-flash')
    parser.add_argument('--deepseek-base-url', default='https://api.deepseek.com')
    parser.add_argument('--deepseek-api-key-env', default='DEEPSEEK_API_KEY')
    parser.add_argument('--timeout-seconds', type=float, default=120)


def add_run_arguments(parser):
    parser.add_argument('run_dirs', nargs='*', type=Path, help='Finished checkpoint run directories')
    parser.add_argument('--run-date', metavar='MMDD', help='Select batch-MMDD-* archives, e.g. 0922')
    parser.add_argument('--checkpoints-root', type=Path, default=ROOT / 'results/checkpoints',
                        help='Search here when --run-date is given without run_dirs')


def select_run_dirs(run_dirs, run_date=None, checkpoints_root=None):
    if run_date is not None and not re.fullmatch(r'\d{4}', run_date):
        raise ValueError('--run-date must be four digits such as 0922')
    if run_dirs:
        candidates = [Path(path) for path in run_dirs]
    elif run_date:
        root = Path(checkpoints_root)
        if not root.is_dir():
            raise ValueError(f'Checkpoint root does not exist: {root}')
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        raise ValueError('Provide run directories or --run-date MMDD')
    if run_date:
        # Use the batch prefix, not another date in the experiment suffix.
        candidates = [path for path in candidates if re.match(rf'^batch-{run_date}-', path.name)]
    selected = list(dict.fromkeys(candidates))
    if not selected:
        raise ValueError(f'No checkpoint runs match batch-{run_date}-' if run_date else 'No run directories')
    missing = [str(path) for path in selected if not path.is_dir()]
    if missing:
        raise ValueError(f'Checkpoint run directory does not exist: {missing[0]}')
    return selected


def load_judge_settings(args):
    selected = args.backend
    config = ({'model': args.local_model, 'base_url': args.local_base_url,
               'ports': args.local_ports, 'api_key': args.local_api_key}
              if selected == 'local_vllm' else
              {'model': args.deepseek_model, 'base_url': args.deepseek_base_url,
               'api_key_env': args.deepseek_api_key_env})
    if not config['model'] or not config['base_url']:
        raise ValueError(f'incomplete {selected} judge configuration')
    url = urlsplit(config['base_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('judge base URL must be HTTP(S) without credentials or query')
    timeout = args.timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('--timeout-seconds must be positive and finite')
    if selected == 'local_vllm':
        try:
            ports = [int(p.strip()) for p in config['ports'].split(',') if p.strip()]
        except ValueError as exc:
            raise ValueError('--local-ports must be comma-separated TCP ports') from exc
        if any(not 1 <= p <= 65535 for p in ports):
            raise ValueError('--local-ports must be valid TCP ports')
        routing = {'model': config['model'], 'base_url': config['base_url'],
                   'load_balancing': {'enabled': bool(ports), 'ports': ports}}
        key = config['api_key']
    else:
        if url.scheme != 'https' or url.hostname != 'api.deepseek.com':
            raise ValueError('DeepSeek API must use https://api.deepseek.com')
        env_name = config['api_key_env']
        if not env_name.isidentifier():
            raise ValueError('invalid DeepSeek API key environment variable')
        routing = {'model': config['model'], 'base_url': config['base_url']}
        key = os.environ.get(env_name)
        if not key:
            from dotenv import dotenv_values
            key = dotenv_values(ROOT / '.env').get(env_name)
    endpoints = resolve_endpoint_urls(routing)
    metadata = {'backend': selected, 'model': config['model'], 'endpoints': endpoints,
                'timeout_seconds': timeout, 'temperature': 0, 'wire_version': 1,
                'config_hash': fingerprint({k: v for k, v in config.items() if k != 'api_key'})}
    if selected == 'deepseek_api':
        metadata.update(api_key_env=env_name, thinking='disabled', response_format='json_object')
    return routing, key, metadata


def make_judge_call(routing, key, metadata):
    if not key:
        raise ValueError(f"Missing judge API key in {metadata.get('api_key_env', 'configuration')}")
    from openai import OpenAI
    endpoints = metadata['endpoints']
    local = threading.local()

    def call(prompt, kind):
        if not hasattr(local, 'clients'):
            local.clients = {url: OpenAI(api_key=key, base_url=url,
                                         timeout=metadata['timeout_seconds'], max_retries=0)
                             for url in endpoints}
        url = next_endpoint(routing, endpoints)
        options = ({'response_format': {'type': 'json_object'},
                    'extra_body': {'thinking': {'type': 'disabled'}}, 'max_tokens': 2048}
                   if metadata['backend'] == 'deepseek_api' else {})
        response = local.clients[url].chat.completions.create(
            model=metadata['model'], messages=[{'role': 'user', 'content': prompt}],
            temperature=0, **options)
        if not response.choices or not isinstance(response.choices[0].message.content, str):
            raise ValueError('empty judge response')
        if response.choices[0].finish_reason == 'length':
            raise ValueError('judge response truncated by model output limit')
        return response.choices[0].message.content.strip()

    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    add_judge_arguments(parser)
    parser.add_argument('--show-prompts', type=int, default=0, metavar='N',
                        help='Print first N actual PTC/Emotion prompt pairs; no model calls or writes')
    parser.add_argument('--ptc-history', type=int, default=6)
    parser.add_argument('--emotion-history', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3, help='Concurrent independent judge calls')
    parser.add_argument('--attempts', type=int, default=3, help='Total attempts per classification per invocation')
    parser.add_argument('--retry-delay', type=float, default=2, help='Exponential backoff base seconds, capped at 60')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='With resume, retry previously failed classifications')
    parser.add_argument('--dry-run', action='store_true', help='Validate extraction without model calls or writes')
    parser.add_argument('--output-root', type=Path, help='Store all run outputs under this separate directory')
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error('workers/attempts must be positive and retry-delay nonnegative')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if min(args.ptc_history, args.emotion_history) < 0 or args.show_prompts < 0:
        parser.error('history limits and --show-prompts must be nonnegative')
    try:
        runs = select_run_dirs(args.run_dirs, args.run_date, args.checkpoints_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Selected {len(runs)} checkpoint run(s):', *[path.name for path in runs], sep='\n  ')
    if args.dry_run or args.show_prompts:
        for path in runs:
            sessions, audit = load_sessions(path)
            print(json.dumps({'run': str(path), 'sessions': len(sessions),
                              'patient_turns': sum(m['role'] == 'patient' for s in sessions for m in s['messages']),
                              'audit': audit}, ensure_ascii=False, indent=2))
            shown = 0
            for session in sessions:
                for index, message in enumerate(session['messages']):
                    if message['role'] != 'patient' or shown >= args.show_prompts:
                        continue
                    for kind, limit in (('ptc', args.ptc_history), ('emotion', args.emotion_history)):
                        print(json.dumps({'run': str(path), 'meeting': session['meeting'], 'kind': kind,
                                          'prompt': make_prompt(kind, session['messages'], index, limit)},
                                         ensure_ascii=False, indent=2))
                    shown += 1
        return 0
    try:
        routing, key, metadata = load_judge_settings(args)
        call = make_judge_call(routing, key, metadata)
    except ValueError as exc:
        parser.error(str(exc))
    print('Judge endpoints:', metadata['endpoints'])
    failed = False
    for path in runs:
        result = run_evaluation(path, call, ptc_history=args.ptc_history,
                                emotion_history=args.emotion_history, workers=args.workers,
                                attempts=args.attempts, retry_delay=args.retry_delay,
                                resume=args.resume, retry_errors=args.retry_errors,
                                metadata=metadata,
                                output_subdir=f"PSI-Bench-style-v2-{metadata['backend']}",
                                output_root=args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failed |= result['status'] != 'complete'
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
