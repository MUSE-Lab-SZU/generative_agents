#!/usr/bin/env python3
"""Gate and preview 0929 Static Profile v6; --execute starts formal judge calls."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanlike_validation.static_profile_recovery import build_pool
from humanlike_validation import static_profile_v6 as study
from runshells.run_psi_bench_eval import add_judge_arguments, load_judge_settings, make_judge_call
from modules.model.endpoint_pool import next_endpoint


def make_static_call(routing, key, metadata):
    """Qwen3 local inference needs thinking disabled for strict JSON outputs."""
    if metadata['backend'] != 'local_vllm':
        return make_judge_call(routing, key, metadata)
    from openai import OpenAI
    local = threading.local()
    endpoints = metadata['endpoints']

    def call(prompt, kind):
        if not hasattr(local, 'clients'):
            local.clients = {url: OpenAI(api_key=key, base_url=url,
                                         timeout=metadata['timeout_seconds'], max_retries=0)
                             for url in endpoints}
        url = next_endpoint(routing, endpoints)
        response = local.clients[url].chat.completions.create(
            model=metadata['model'], messages=[{'role': 'user', 'content': prompt}],
            temperature=0, max_tokens=2048,
            response_format={'type': 'json_object'},
            extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        if not response.choices or response.choices[0].finish_reason == 'length':
            raise ValueError('empty or truncated local Qwen response')
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError('empty local Qwen response')
        return content.strip()

    return call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-root', type=Path, default=ROOT / 'results/experiment_data/batch_state')
    parser.add_argument('--checkpoints-root', type=Path, default=ROOT / 'results/checkpoints')
    parser.add_argument('--agents-dir', type=Path, default=ROOT / 'frontend/static/assets/village/agents')
    parser.add_argument('--fact-table', type=Path, default=study.FACT_PATH)
    parser.add_argument('--output', type=Path, default=ROOT / 'humanlike_outputs/0929_static_profile_v6')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--show-prompts', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reassess-cache', action='store_true',
                        help='With --resume, reparse cached raw responses using current evidence rules')
    parser.add_argument('--execute', action='store_true', help='After completion gate, call configured judge and write v6 outputs')
    add_judge_arguments(parser)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.bootstraps < 0 or args.show_prompts < 0:
        parser.error('workers must be positive; bootstraps and show-prompts nonnegative')
    if args.reassess_cache and not args.resume:
        parser.error('--reassess-cache requires --resume')
    try:
        # No checkpoint is opened until all 15 batch states say completed.
        plan, _ = study.discover_complete_study(args.state_root, args.checkpoints_root)
        fact_table = study.load_fact_table(args.fact_table, args.agents_dir)
        pool = build_pool(args.agents_dir)
        samples = study.prepare_study(plan, args.checkpoints_root, pool, fact_table, args.seed)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    preview = {'status': 'preflight_passed', 'runs': len(plan), 'sessions': len(samples) // 2,
               'questions': len(samples), 'conditions': {condition: 5 for condition in study.CONDITIONS},
               'persona': fact_table['persona'], 'mode': 'single-session',
               'fact_table_hash': study.fingerprint(fact_table)}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    for sample in samples[:args.show_prompts]:
        print(json.dumps({'condition': sample['condition'], 'repeat': sample['repeat'],
                          'session_number': sample['session_number'], 'field': sample['field'],
                          'prompt': sample['prompt']}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error(f'output already exists: {args.output}; use --resume for identical inputs')
    try:
        routing, key, metadata = load_judge_settings(args)
        if metadata['backend'] == 'local_vllm':
            metadata = {**metadata, 'thinking': 'disabled', 'max_tokens': 2048,
                        'response_format': 'json_object'}
        call = make_static_call(routing, key, metadata)
        _, manifest = study.run_cached(args.output, samples, fact_table, plan, call, metadata,
                                       workers=args.workers, resume=args.resume,
                                       reassess_cache=args.reassess_cache,
                                       bootstraps=args.bootstraps, seed=args.seed)
    except (OSError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output': str(args.output), 'status': manifest['status'],
                      'errors': manifest['errors']}, ensure_ascii=False))
    return int(manifest['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
