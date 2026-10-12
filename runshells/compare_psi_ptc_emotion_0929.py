#!/usr/bin/env python3
"""Preview or execute the 0929 ESC-only repeated-run PTC/Emotion validation.

The completion gate is mandatory for both modes. Default mode previews the
sample without model calls or writes. --execute labels only after all 15 runs
and 180 CBT sessions are confirmed complete.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from humanlike_validation import psi_bench_eval as psi
from humanlike_validation.psi_bench_eval import (
    COMPARISON_PROMPTS, analyze_clustered, eligible_patient_session,
    make_comparison_prompt, merge, patient_turn_rows, select_aligned_external,
)
from runshells.run_psi_bench_eval import add_judge_arguments, load_judge_settings, make_psi_label_call

PROMPTS = COMPARISON_PROMPTS
prompt = make_comparison_prompt


CONDITIONS = ('G1', 'G4', 'G7')
BASELINES = ('PatientPSI', 'RoleplayDoh')
REAL = 'ESC real'
TURN_COUNT = 8
SESSION_COUNT = 12
REPEAT_COUNT = 5
CONDITION_PATTERN = re.compile(r'^Counsel-([^-]+)-(G1|G4|G7)-(MILD|MOD|SEV)(?:--|$)')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def discover_runs(state_root, checkpoints_root):
    """Check all 15 completed batch states before any checkpoint is opened."""
    state_root, checkpoints_root = Path(state_root), Path(checkpoints_root)
    problems, found = [], {}
    for repeat in range(1, REPEAT_COUNT + 1):
        batch = f'batch-0929-{repeat:02d}'
        folder = state_root / batch
        states = [] if not folder.is_dir() else sorted(folder.glob('*.json'))
        for path in states:
            if path.name in ('manifest.json',):
                continue
            try:
                state = _json(path)
            except (OSError, ValueError) as exc:
                problems.append(f'{batch}/{path.name}: unreadable state: {exc}')
                continue
            if not isinstance(state, dict) or state.get('batch_name') != batch:
                continue
            group = str(state.get('group', '')).upper()
            if group not in CONDITIONS:
                continue
            key = (group, repeat)
            if key in found:
                problems.append(f'{batch}/{group}: duplicate condition state')
                continue
            found[key] = (state, path)
        for group in CONDITIONS:
            key = (group, repeat)
            if key not in found:
                problems.append(f'{batch}/{group}: missing condition state')
                continue
            state, path = found[key]
            run_name = state.get('run_name', '')
            if state.get('status') != 'completed' or state.get('last_completed_phase') != 'completed':
                problems.append(f'{batch}/{group}: status={state.get("status")}, phase={state.get("last_completed_phase")}')
            condition_name = str(state.get('condition_name') or '')
            if not isinstance(run_name, str) or not run_name.startswith(f'{batch}-{condition_name}-'):
                problems.append(f'{batch}/{group}: invalid run_name={run_name!r}')
            elif not (checkpoints_root / run_name).is_dir():
                problems.append(f'{batch}/{group}: missing checkpoint directory {run_name}')
            match = CONDITION_PATTERN.match(condition_name)
            valid_g4 = group == 'G4' and bool(re.fullmatch(r'Counsel-G4-(MILD|MOD|SEV)', condition_name))
            if not (match and match.group(2) == group) and not valid_g4:
                problems.append(f'{batch}/{group}: invalid condition identity in {path.name}')
    if problems:
        raise ValueError('0929 run completion gate failed:\n- ' + '\n- '.join(problems))
    if len({state['run_name'] for state, _ in found.values()}) != len(found):
        raise ValueError('0929 run completion gate failed: run names are not 15 independent archives')
    identities = {(state.get('variant'), state.get('severity')) for state, _ in found.values()}
    if len(identities) != 1 or any(not all(x) for x in identities):
        raise ValueError(f'0929 requires one persona/severity across 15 runs; found {sorted(map(str, identities))}')
    return [{'condition': group, 'repeat': repeat,
             'run': found[(group, repeat)][0]['run_name'],
             'state_file': str(found[(group, repeat)][1].resolve()),
             'persona': next(iter(identities))[0], 'severity': next(iter(identities))[1]}
            for group in CONDITIONS for repeat in range(1, REPEAT_COUNT + 1)]


def load_ours(run_plan, checkpoints_root, turns=TURN_COUNT):
    """Require 12 distinct completed meetings; number them in time order."""
    selected, problems, coverage = [], [], []
    for item in run_plan:
        run_path = Path(checkpoints_root) / item['run']
        try:
            sessions, audit = psi.load_sessions(run_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{item['run']}: cannot load completed checkpoint: {exc}")
            continue
        if audit:
            problems.append(f"{item['run']}: {len(audit)} consult record/mapping errors: {audit[:3]}")
        meetings = [session['meeting'] for session in sessions]
        if len(sessions) != SESSION_COUNT or len(set(meetings)) != SESSION_COUNT:
            problems.append(f"{item['run']}: expected {SESSION_COUNT} distinct completed meetings, "
                            f"found {len(sessions)} records / {len(set(meetings))} meeting IDs")
        for number, session in enumerate(sessions, start=1):
            admitted = eligible_patient_session(session['messages'], turns)
            count = sum(m['role'] == 'patient' for m in merge(session['messages']))
            eligible = admitted is not None
            coverage.append({**item, 'session_number': number, 'meeting': session['meeting'],
                             'cbt_session': session['cbt_session'],
                             'full_patient_turns': count, 'eligible': eligible})
            if eligible:
                selected.append({**item, 'dataset': item['condition'],
                                 'id': f"{item['run']}:{session['meeting']}",
                                 'session_number': number, 'meeting': session['meeting'],
                                 'cbt_session': session['cbt_session'],
                                 'source': 'ours', 'language': 'zh',
                                 'origin': session['source'], 'full_patient_turns': count,
                                 'messages': admitted[0]})
    if problems:
        raise ValueError('0929 session completeness gate failed:\n- ' + '\n- '.join(problems))
    for group in CONDITIONS:
        if not any(s['dataset'] == group for s in selected):
            raise ValueError(f'{group}: no sessions with at least {turns} patient turns')
    return selected, coverage


def _number(value):
    return 'NA' if value is None else f'{value:.3f}'


def report(summary, coverage, external, manifest):
    groups = summary['groups']
    judge = summary['judge']
    persona = manifest['run_plan'][0]['persona']
    lines = [
        '# 0929 PSI-Bench PTC/Emotion 类人验证', '',
        '## 方法与数据规模', '',
        '主 reference 固定为 Eeyore ESC。比较量是同轮患者回复标签分布的 SciPy Jensen–Shannon distance（自然对数底）再对前 8 轮等权平均。'
        '本项目与 ESC 无同病例配对，按 external reference distribution comparison 解释。PatientPSI、RoleplayDoh 均限定 source=esc、'
        'backend_llm=gpt-4.1-mini，并与 ESC 使用同一批原始行索引/session_id。所有组至少 8 条患者回复，统一截取前 8 条。',
        f'本项目 persona 为 `{persona}`。自动标注统一使用本地 `{judge["model"]}`（{judge["backend"]}），温度 0、关闭思考、JSON 枚举约束；'
        f'共 {sum(group["patient_turns"] for group in groups.values())} 条患者回复、'
        f'{2 * sum(group["patient_turns"] for group in groups.values())} 个 PTC/Emotion 标签，分类错误为 0。', '',
        '| 条件 | 独立 repeats | 完整 sessions | 有效 sessions | 有效患者轮 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for condition in CONDITIONS:
        entries = [c for c in coverage if c['condition'] == condition]
        lines.append(f"| {condition} | {len(set(c['repeat'] for c in entries))} | {len(entries)} | "
                     f"{groups[condition]['sessions']} | {groups[condition]['patient_turns']} |")
    for name in (REAL, *BASELINES):
        info = external[name]
        lines.append(f"| {name} | — | {info['candidate']} 候选 | {groups[name]['sessions']} "
                     f"(合格 {info['eligible']}) | {groups[name]['patient_turns']} |")
    lines += [
        '', f"外部三组共同合格 ID 为 {external['common_eligible_ids']}；按预先固定 seed={manifest['selection_seed']} "
        f"选择 {external['selected_ids']}，目标数为三个条件有效 session 数的中位数 "
        f"{external['target_ids']}。不足时使用实际数量，不重复会谈。",
        '同一 persona 在每个 condition 下有 5 次独立 repeat；每次 12 场完整 Session。有效场数只指满足前 8 患者轮准入的场次，'
        '同一场内回复不视为独立患者。会谈序号按完成记录的时间顺序确定，原始 `cbt_session` 阶段标签另存于 manifest 和逐项结果。', '',
        '## PTC / Emotion 全类别分布', '',
    ]
    for kind, labels in psi.LABELS.items():
        lines += [f'### {kind.upper()}', '',
                  '| 组别 | ' + ' | '.join(labels) + ' |',
                  '| --- | ' + ' | '.join('---:' for _ in labels) + ' |']
        for name in (REAL, *CONDITIONS, *BASELINES):
            lines.append('| ' + name + ' | ' + ' | '.join(_number(groups[name][kind][label]) for label in labels) + ' |')
        lines.append('')
    lines += ['## 主结果：与同一 ESC reference 的逐轮平均 JS distance', '',
              '| 组别 | PTC 全四类 [95% CI] | Emotion 全九类 [95% CI] |',
              '| --- | ---: | ---: |']
    for name in (*CONDITIONS, *BASELINES):
        cells = []
        for kind in psi.LABELS:
            item = summary['comparisons'][name][kind]
            ci = item['ci95']
            cells.append(f"{_number(item['mean'])} [{_number(ci['lower'])}, {_number(ci['upper'])}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ['', '下表为同一次 bootstrap 内共享 ESC reference 和外部 session ID 的差值（本项目条件减公开基线）；'
              '正值表示距离更大。区间只描述本次固定 persona 与选样。', '',
              '| 差值 | PTC [95% CI] | Emotion [95% CI] |', '| --- | ---: | ---: |']
    for condition in CONDITIONS:
        for baseline in BASELINES:
            cells = []
            for kind in psi.LABELS:
                item = summary['differences'][condition][baseline][kind]
                ci = item['ci95']
                cells.append(f"{_number(item['estimate'])} [{_number(ci['lower'])}, {_number(ci['upper'])}]")
            lines.append(f"| {condition} − {baseline} | {' | '.join(cells)} |")
    ptc_patient = [summary['differences'][condition]['PatientPSI']['ptc']['ci95'] for condition in CONDITIONS]
    emotion_patient = [summary['differences'][condition]['PatientPSI']['emotion']['ci95'] for condition in CONDITIONS]
    roleplay_bounds = [summary['differences'][condition]['RoleplayDoh'][kind]['ci95']
                       for condition in CONDITIONS for kind in psi.LABELS]
    lines += ['', '### 结果解读', '']
    if all(bound['lower'] > 0 for bound in ptc_patient):
        lines.append('PTC 全四类下，G1/G4/G7 与 ESC 的距离均高于 PatientPSI；三个差值的 95% bootstrap 区间均高于 0。')
    if emotion_patient[0]['upper'] < 0:
        lines.append('Emotion 全九类下，G1 与 ESC 的距离低于 PatientPSI，G1 − PatientPSI 的区间低于 0；'
                     'G4/G7 的相应区间跨 0。')
    if all(bound['lower'] <= 0 <= bound['upper'] for bound in roleplay_bounds):
        lines.append('G1/G4/G7 与 RoleplayDoh 的 PTC 和 Emotion 差值区间均跨 0。')
    lines.append('这些是指定 ESC 样本和自动标签分布的距离比较，不能解释为模拟器优劣或治疗效果。')
    lines.append('分布上，ESC 的 P 标签比例为 {:.1%}，G1/G4/G7 分别为 {:.1%}/{:.1%}/{:.1%}；'
                 'ESC 的 neutral 比例为 {:.1%}，G1/G4/G7 分别为 {:.1%}/{:.1%}/{:.1%}。'.format(
                     groups[REAL]['ptc']['P'], *(groups[name]['ptc']['P'] for name in CONDITIONS),
                     groups[REAL]['emotion']['neutral'],
                     *(groups[name]['emotion']['neutral'] for name in CONDITIONS)))
    lines += ['', '## 补充敏感性分析', '',
              '去 F / 去 neutral 是条件于剩余类别的分布比较；零剩余质量的轮次被排除，具体有效轮次见 summary.json。'
              '前 4 / 前 6 轮也仅作为补充。', '',
              '| 组别 | PTC 去 F | Emotion 去 neutral | PTC 前 4 / 6 轮 | Emotion 前 4 / 6 轮 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in (*CONDITIONS, *BASELINES):
        item = summary['supplementary_sensitivity'][name]
        lines.append(f"| {name} | {_number(item['ptc']['without_filler_or_neutral']['mean'])} | "
                     f"{_number(item['emotion']['without_filler_or_neutral']['mean'])} | "
                     f"{_number(item['ptc']['first_4']['mean'])} / {_number(item['ptc']['first_6']['mean'])} | "
                     f"{_number(item['emotion']['first_4']['mean'])} / {_number(item['emotion']['first_6']['mean'])} |")
    lines += ['', '## 与历史探索版的差异', '',
              '0923 使用 ESConv/AnnoMI 独立参考和本地 Qwen；0928 使用 ESC+HOPE+AnnoMI 的 Eeyore mixed real、'
              '一个 0922 checkpoint 和 DeepSeek。旧产物保留为 supplementary / reference sensitivity，不覆盖、不与本次数字相减或据 JSD 大小更换主分析。'
              '本次固定 ESC-only、同后端同 source 的两个公开 baseline、共同 ESC ID、三条件各 5 次 repeat、分层 bootstrap。', '',
              '## 不确定性与限制', '',
              f"bootstrap 使用 {summary['design']['bootstrap_draws']} 次有放回抽样：本项目 repeat → session，"
              '每场前 8 条患者回复成簇保留；ESC 与两个 baseline 按共同 session ID 成组抽样。95% CI 为百分位区间。'
              '当前只有一个 persona，因此 CI 只反映该 persona 下独立 repeats/session 的不确定性，不能解释为跨 persona 泛化。', '',
              '本项目为中文 CBT 医患仿真，ESC 和公开基线为英文且场景、回复长度、治疗师流程和生成后端不完全一致；'
              '自动标签可能有跨语言误差，未有人类金标准。Eeyore ESC 是公开对话参考分布，不能视为临床真实患者金标准；'
              'PTC 的 C 和 Emotion 标签不代表治疗效果。本项目和 ESC 没有同病例配对，也不能由分布距离推出因果或方法优劣。', '',
              '## 复核产物与图表选择', '',
              'sampling_manifest.json 记录入样与完整性证据；turns.jsonl 保存逐项文本、层级 ID、标签和状态；'
              'bootstrap.jsonl 保存逐次抽样统计；summary.json 保存所有分布、逐轮距离和 CI。'
              'ptc_emotion_main.png 是唯一主图：上排全类比例，下排主 JS distance 与 95% CI。', '',
              '未生成逐 repeat 图（单个 repeat 不能代表条件）、逐轮单独大批量图（与主图/JSON 重复）、'
              '跨 persona 图（仅一个 persona）、长度/翻译匹配图（缺少人工校验与可比字段）、'
              'HOPE/AnnoMI mixed real 新结果图（本次只预设 ESC 主分析）。历史 mixed real 图与数值保持原址。', ''
    ]
    return '\n'.join(lines)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = (REAL, *CONDITIONS, *BASELINES)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (kind, labels) in enumerate(psi.LABELS.items()):
        ax = axes[0, column]
        starts = [0.0] * len(names)
        for label in labels:
            values = [summary['groups'][name][kind][label] for name in names]
            ax.barh(names, values, left=starts, label=label)
            starts = [a + b for a, b in zip(starts, values)]
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f'{kind.upper()} distribution (all labels)')
        ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.06))
        ax = axes[1, column]
        compared = (*CONDITIONS, *BASELINES)
        values = [summary['comparisons'][name][kind]['mean'] for name in compared]
        lower = [max(0, value - summary['comparisons'][name][kind]['ci95']['lower'])
                 for name, value in zip(compared, values)]
        upper = [max(0, summary['comparisons'][name][kind]['ci95']['upper'] - value)
                 for name, value in zip(compared, values)]
        ax.barh(compared, values, color=['#3b82f6'] * 3 + ['#64748b'] * 2)
        ax.errorbar(values, range(len(compared)), xerr=[lower, upper], fmt='none', ecolor='#111827', capsize=3)
        ax.set_xlim(0, .85)
        ax.invert_yaxis()
        ax.set_title(f'{kind.upper()} mean per-turn JS distance to ESC (95% CI)')
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def evaluate(args, selected, manifest):
    routing, key, metadata = load_judge_settings(args)
    call = make_psi_label_call(routing, key, metadata)
    signature = psi.fingerprint({'selected': selected, 'prompts': PROMPTS,
                                 'turns': TURN_COUNT, 'judge': metadata})
    output = args.output
    manifest_path = output / 'sampling_manifest.json'
    if manifest_path.exists():
        old = psi.read_json(manifest_path)
        if old.get('signature') != signature:
            raise ValueError('Output contains a different sample/prompt/judge; choose another --output')
    manifest.update(signature=signature, judge=metadata, status='labelling')
    psi.atomic_json(manifest_path, manifest)
    rows = patient_turn_rows(selected)
    sessions_by_id = {(session['dataset'], session['id']): session for session in selected}
    cache = output / 'classifications'
    cache.mkdir(exist_ok=True)
    tasks = []
    for row in rows:
        session = sessions_by_id[(row['dataset'], row['id'])]
        for kind in psi.LABELS:
            tasks.append((row, kind, prompt(kind, session['messages'], row['message_index'])))

    def work(task):
        row, kind, body = task
        path = cache / (psi.fingerprint([signature, row['dataset'], row['id'], row['turn_index'], kind]) + '.json')
        saved = psi.read_json(path) if path.exists() else None
        if saved and saved.get('status') == 'ok':
            return row, kind, saved
        result = psi.classify(call, kind, body)
        psi.atomic_json(path, result)
        return row, kind, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, kind, result in pool.map(work, tasks):
            for key in ('label', 'status', 'error'):
                row[f'{kind}_{key}'] = result.get(key)
    with (output / 'turns.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    errors = sum(row[f'{kind}_status'] != 'ok' for row in rows for kind in psi.LABELS)
    if errors:
        psi.atomic_json(manifest_path, {**manifest, 'status': 'classification_errors', 'errors': errors})
        raise RuntimeError(f'{errors} classification errors; inspect turns.jsonl and rerun to retry')
    summary, draws = analyze_clustered(
        rows, conditions=CONDITIONS, baselines=BASELINES, reference=REAL,
        turns=TURN_COUNT, bootstraps=args.bootstraps, seed=args.seed,
        expected_repeats=REPEAT_COUNT, persona_count=1)
    summary['design']['ci_scope'] = 'conditional on one persona; independent repeats and sessions only'
    summary['judge'] = metadata
    psi.atomic_json(output / 'summary.json', summary)
    with (output / 'bootstrap.jsonl').open('w', encoding='utf-8') as stream:
        for draw in draws:
            stream.write(json.dumps(draw, ensure_ascii=False) + '\n')
    (output / 'REPORT.md').write_text(report(summary, manifest['ours_coverage'], manifest['external_coverage'], manifest), encoding='utf-8')
    plot(summary, output / 'ptc_emotion_main.png')
    psi.atomic_json(manifest_path, {**manifest, 'status': 'complete', 'labelled_turns': len(rows), 'errors': 0})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_judge_arguments(parser)
    parser.add_argument('--state-root', type=Path, default=ROOT/'results/experiment_data/batch_state')
    parser.add_argument('--checkpoints-root', type=Path, default=ROOT/'results/checkpoints')
    parser.add_argument('--eeyore-parquet', type=Path,
                        default=ROOT/'data/external/eeyore_profile/data/train-00000-of-00001.parquet')
    parser.add_argument('--synthetic-parquet', type=Path,
                        default=ROOT/'data/external/psibench-PsyCoPref/train.parquet')
    parser.add_argument('--output', type=Path, default=ROOT/'humanlike_outputs/psi_ptc_emotion_0929_esc')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--execute', action='store_true', help='Call configured judge after mandatory completeness checks')
    args = parser.parse_args()
    if args.workers < 1 or args.bootstraps < 1:
        parser.error('workers and bootstraps must be positive')
    try:
        plan = discover_runs(args.state_root, args.checkpoints_root)
        ours, coverage = load_ours(plan, args.checkpoints_root)
        counts = [sum(s['dataset'] == condition for s in ours) for condition in CONDITIONS]
        target = sorted(counts)[1]  # Shared ESC sample closest to the three condition sample sizes.
        external, external_coverage = select_aligned_external(
            args.eeyore_parquet, args.synthetic_parquet, target=target, seed=args.seed,
            turns=TURN_COUNT, reference=REAL,
            simulators={'patientpsi': 'PatientPSI', 'roleplaydoh': 'RoleplayDoh'})
        selected = ours + external
        manifest = {'schema_version': 1, 'study': '0929 ESC-only PTC/Emotion',
                    'selection_seed': args.seed, 'turns': TURN_COUNT,
                    'run_plan': plan, 'ours_coverage': coverage,
                    'external_coverage': external_coverage,
                    'selected': [{key: value for key, value in session.items() if key != 'messages'}
                                 for session in selected],
                    'session_content_sha256': psi.fingerprint(selected)}
        if not args.execute:
            print(json.dumps({'gate': 'passed', 'runs': len(plan), 'complete_sessions': len(coverage),
                              'eligible_ours_by_condition': dict(zip(CONDITIONS, counts)),
                              'external': external_coverage,
                              'message': 'Preparation only; no judge calls or output files'}, ensure_ascii=False, indent=2))
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        evaluate(args, selected, manifest)
        print(f'Complete: {args.output / "REPORT.md"}')
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
