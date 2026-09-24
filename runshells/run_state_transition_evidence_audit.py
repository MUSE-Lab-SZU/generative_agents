#!/usr/bin/env python3
"""Offline State Transition Evidence Audit for finished V3 checkpoint archives."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model.state_transition_evidence_audit import (
    digest, parse_rating, prepare, read_json, score, summarize,
)
from runshells.run_psi_bench_eval import add_judge_arguments, load_judge_settings, make_judge_call


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    os.replace(tmp, path)


def call_rating(call, prompt, kind, sample, attempts, retry_delay):
    raw = None
    for attempt in range(attempts):
        try:
            raw = call(prompt, kind)
            return {'status': 'ok', 'rating': parse_rating(raw, kind, sample), 'raw': raw, 'error': None}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if attempt + 1 < attempts:
                time.sleep(min(retry_delay * 2 ** attempt, 60))
    return {'status': 'error', 'rating': None, 'raw': raw, 'error': error}


def evaluate(run, patient, args, call, metadata):
    samples = prepare(run, patient)
    if metadata["backend"] == "local_vllm" and metadata["model"].lower().startswith("qwen3"):
        for sample in samples:
            sample["evidence_prompt"] += "\n/nothink"
            sample["state_prompt"] += "\n/nothink"
    output_root = args.output_root or ROOT / 'results' / 'experiment_data'
    output = output_root / run.name / f'state_transition_evidence_audit_v1_{metadata["backend"]}'
    manifest_path = output / 'manifest.json'
    rows_path = output / 'items.jsonl'
    identity = digest({'samples': samples, 'metadata': metadata, 'patient': patient})
    if output.exists() and not args.resume:
        raise ValueError(f'Output already exists: {output}; use --resume')
    if args.resume and output.exists():
        if not manifest_path.is_file() or read_json(manifest_path).get('identity') != identity:
            raise ValueError(f'Resume identity mismatch: {output}')
    output.mkdir(parents=True, exist_ok=True)
    existing = {}
    sample_by_id = {sample["id"]: sample for sample in samples}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('status') == 'error' and row.get('id') in sample_by_id:
                sample = sample_by_id[row['id']]
                for kind in ('evidence', 'state'):
                    if row.get(f'{kind}_rating') is None and row.get(f'{kind}_raw'):
                        try:
                            row[f'{kind}_rating'] = parse_rating(row[f'{kind}_raw'], kind, sample)
                            row[f'{kind}_error'] = None
                        except Exception:
                            pass
                if row.get('evidence_rating') and row.get('state_rating'):
                    row['status'] = 'ok'
            row = score(row)
            if row['id'] in existing:
                raise ValueError(f'Duplicate cached item: {row["id"]}')
            existing[row['id']] = row
    manifest = {'schema_version': 1, 'identity': identity, 'status': 'running',
                'run': str(run.resolve()), 'patient': patient, 'model': metadata,
                'source_snapshot': samples[0]['source_snapshot'] if samples else None,
                'n_decisions': len(samples), 'n_changed': sum(s['changed'] for s in samples),
                'prompt_hashes': {'evidence': digest(samples[0]['evidence_prompt'].split('数据：\n')[0]) if samples else None,
                                  'state': digest(samples[0]['state_prompt'].split('节点：\n')[0]) if samples else None}}
    atomic_json(manifest_path, manifest)
    def evaluate_one(sample):
        er = call_rating(call, sample['evidence_prompt'], 'evidence', sample, args.attempts, args.retry_delay)
        if sample['changed']:
            sr = call_rating(call, sample['state_prompt'], 'state', sample, args.attempts, args.retry_delay)
        else:
            sr = {'status': 'ok', 'rating': {'direction': 'unchanged', 'degree': 0,
                                           'reason': 'No committed graph change'}, 'raw': None, 'error': None}
        return score({**sample, 'evidence_rating': er['rating'], 'state_rating': sr['rating'],
                      'evidence_raw': er['raw'], 'state_raw': sr['raw'],
                      'evidence_error': er['error'], 'state_error': sr['error'],
                      'status': 'ok' if er['status'] == sr['status'] == 'ok' else 'error'})

    pending = [s for s in samples if s['id'] not in existing or
               (args.retry_errors and existing[s['id']].get('status') != 'ok')]
    done = {key: value for key, value in existing.items() if key not in {s['id'] for s in pending}}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, sample): sample for sample in pending}
        for completed in as_completed(futures):
            sample = futures[completed]
            row = completed.result()
            done[sample['id']] = row
            ordered = [done[s['id']] for s in samples if s['id'] in done]
            atomic_jsonl(rows_path, ordered)
            print(f'{run.name}: {len(done)}/{len(samples)} {sample["id"]} {row["support_status"]}', flush=True)
    ordered = [done[s['id']] for s in samples]
    atomic_jsonl(rows_path, ordered)
    summary = summarize(ordered)
    summary['run'] = str(run.resolve())
    summary['patient'] = patient
    summary['errors'] = sum(r['status'] != 'ok' for r in ordered)
    atomic_json(output / 'summary.json', summary)
    manifest['status'] = 'complete' if not summary['errors'] else 'partial'
    manifest['errors'] = summary['errors']
    atomic_json(manifest_path, manifest)
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path, help='Finished results/checkpoints/<run> archives')
    parser.add_argument('--patient', default='卡布达')
    parser.add_argument('--output-root', type=Path, help='Default: results/experiment_data')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-delay', type=float, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate archive and count decisions, no model calls or writes')
    parser.add_argument('--show-prompts', type=int, default=0, metavar='N', help='Print first N blind prompts; no model calls or writes')
    add_judge_arguments(parser)
    args = parser.parse_args()
    if args.attempts < 1 or args.workers < 1 or args.retry_delay < 0 or args.show_prompts < 0:
        parser.error('Invalid attempts/retry-delay/show-prompts')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    runs = [p.resolve() for p in args.run_dirs]
    if len(set(runs)) != len(runs):
        parser.error('Duplicate run directories')
    if args.dry_run or args.show_prompts:
        for run in runs:
            samples = prepare(run, args.patient)
            print(json.dumps({'run': str(run), 'decisions': len(samples),
                              'changed': sum(s['changed'] for s in samples),
                              'sessions': len({s['session_id'] for s in samples})}, ensure_ascii=False))
            for item in samples[:args.show_prompts]:
                print(json.dumps({'id': item['id'], 'evidence_prompt': item['evidence_prompt'],
                                  'state_prompt': item['state_prompt']}, ensure_ascii=False, indent=2))
        return 0
    try:
        routing, key, metadata = load_judge_settings(args)
        call = make_judge_call(routing, key, metadata)
    except ValueError as exc:
        parser.error(str(exc))
    failed = False
    for run in runs:
        output, summary = evaluate(run, args.patient, args, call, metadata)
        print(json.dumps({'output': str(output), 'summary': summary}, ensure_ascii=False, indent=2))
        failed |= bool(summary['errors'])
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
