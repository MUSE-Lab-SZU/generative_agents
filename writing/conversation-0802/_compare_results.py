#!/usr/bin/env python3
"""
对比分析脚本：将心理老师的专业判断与系统量表（PHQ-9 / BDI-II）分数进行对比。

使用流程：
  1. 老师完成 Word 文档中的评估表
  2. 将老师的判断填入 teacher_judgments.json（见下方模板）
  3. 运行本脚本：python _compare_results.py
  4. 查看输出的对比报告

输入文件：
  - teacher_judgments.json  老师对每个案例的判断
  - *_summary.md            系统量表评估汇总

输出：
  - comparison_report.md    对比分析报告
"""

import json
import os
import re
import glob
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PHQ-9 / BDI-II severity classification
# ============================================================

PHQ9_CLASSIFICATION = [
    (0, 4, "无抑郁"),
    (5, 9, "轻度抑郁"),
    (10, 14, "中度抑郁"),
    (15, 19, "中重度抑郁"),
    (20, 27, "重度抑郁"),
]

BDI2_CLASSIFICATION = [
    (0, 13, "无抑郁"),
    (14, 19, "轻度抑郁"),
    (20, 28, "中度抑郁"),
    (29, 63, "重度抑郁"),
]


def classify_phq9(score):
    """Classify PHQ-9 score into severity level."""
    for lo, hi, label in PHQ9_CLASSIFICATION:
        if lo <= score <= hi:
            return label
    return "未知"


def classify_bdi2(score):
    """Classify BDI-II score into severity level."""
    for lo, hi, label in BDI2_CLASSIFICATION:
        if lo <= score <= hi:
            return label
    return "未知"


def severity_rank(label):
    """Convert severity label to numeric rank for comparison."""
    labels = ["无抑郁", "轻度抑郁", "中度抑郁", "中重度抑郁", "重度抑郁"]
    try:
        return labels.index(label)
    except ValueError:
        # Handle combined labels like "并列（中度抑郁/中重度抑郁）"
        if "中度抑郁" in label and "中重度抑郁" in label:
            return 2.5
        if "轻度抑郁" in label and "中度抑郁" in label:
            return 1.5
        if "中度抑郁" in label and "重度抑郁" in label:
            return 2.5
        return -1


# ============================================================
# Parse summary MD files
# ============================================================

def parse_summary_md(filepath):
    """
    Parse a summary MD file and extract PHQ-9 and BDI-II scores.
    Returns dict:
      {
        'condition': 'Counsel-KBD1-G1-SEV',
        'variant': 'kbd1',
        'group': 'g1',
        'severity': 'severe',
        'run_name': '...',
        'phq9': [{'point': 'T0', 'score': 19.8, 'severity': '中重度抑郁'}, ...],
        'bdi2': [{'point': 'T0', 'score': 43.5, 'severity': '重度抑郁'}, ...],
      }
    """
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    result = {
        'condition': '',
        'variant': '',
        'group': '',
        'severity': '',
        'run_name': '',
        'phq9': [],
        'bdi2': [],
    }

    # Extract condition from the first heading
    m = re.search(r'# .*?(Counsel-\w+)', content)
    if m:
        result['condition'] = m.group(1)

    # Extract variant, group, severity from the "主结果" section
    m = re.search(r'### (Counsel-\S+)', content)
    if m:
        result['condition'] = m.group(1)

    m = re.search(r'- variant:\s*`(\w+)`', content)
    if m:
        result['variant'] = m.group(1)

    m = re.search(r'- group:\s*`(\w+)`', content)
    if m:
        result['group'] = m.group(1).replace('g', '')  # normalize: 'g1' -> '1'

    m = re.search(r'- severity:\s*`(\w+)`', content)
    if m:
        result['severity'] = m.group(1)

    m = re.search(r'- run_name:\s*`([^`]+)`', content)
    if m:
        result['run_name'] = m.group(1)

    # Parse PHQ-9 table
    phq9_section = re.search(r'#### PHQ-9\n\n(.*?)(?:\n####|\Z)', content, re.DOTALL)
    if phq9_section:
        for line in phq9_section.group(1).split('\n'):
            m = re.match(
                r'\|\s*(\S+)\s*\|\s*\d+\s*\|\s*\S+\s*\|\s*([\d.]+)\s*\|\s*(.+?)\s*\|',
                line
            )
            if m:
                result['phq9'].append({
                    'point': m.group(1),
                    'score': float(m.group(2)),
                    'severity': m.group(3).strip(),
                })

    # Parse BDI-II table
    bdi2_section = re.search(r'#### BDI-II\n\n(.*?)(?:\n##|\Z)', content, re.DOTALL)
    if bdi2_section:
        for line in bdi2_section.group(1).split('\n'):
            m = re.match(
                r'\|\s*(\S+)\s*\|\s*\d+\s*\|\s*\S+\s*\|\s*([\d.]+)\s*\|\s*(.+?)\s*\|',
                line
            )
            if m:
                result['bdi2'].append({
                    'point': m.group(1),
                    'score': float(m.group(2)),
                    'severity': m.group(3).strip(),
                })

    return result


# ============================================================
# Load all summary data
# ============================================================

def load_all_summaries():
    """Load all summary MD files in the current directory."""
    summaries = {}
    for f in sorted(glob.glob('*_summary.md')):
        data = parse_summary_md(f)
        # Use variant as key
        variant = data['variant']
        group = data['group']  # already normalized in parse_summary_md
        run = '01' if '-0802-01_' in f else '02' if '-0802-02_' in f else '??'
        key = f"{variant}-g{group}-{run}"
        summaries[key] = data
        print(f"  [LOAD] {f} -> {key}")
    return summaries


# ============================================================
# Comparison logic
# ============================================================

def compare_severity_levels(teacher_label, system_score, scale='phq9'):
    """Compare teacher's severity judgment with system score classification."""
    if scale == 'phq9':
        system_label = classify_phq9(system_score)
    else:
        system_label = classify_bdi2(system_score)

    teacher_rank = severity_rank(teacher_label)
    system_rank = severity_rank(system_label)

    exact_match = (teacher_label == system_label)

    # Adjacent match (within 1 rank level)
    if teacher_rank >= 0 and system_rank >= 0:
        rank_diff = abs(teacher_rank - system_rank)
        adjacent_match = rank_diff <= 0.5  # within same level or adjacent
    else:
        adjacent_match = False

    return {
        'teacher_label': teacher_label,
        'system_label': system_label,
        'system_score': system_score,
        'exact_match': exact_match,
        'adjacent_match': adjacent_match,
        'rank_diff': abs(teacher_rank - system_rank) if (teacher_rank >= 0 and system_rank >= 0) else None,
    }


def check_score_in_range(teacher_range_lo, teacher_range_hi, system_score):
    """Check if system score falls within teacher's estimated range."""
    try:
        lo = float(teacher_range_lo) if teacher_range_lo else None
        hi = float(teacher_range_hi) if teacher_range_hi else None
        if lo is not None and hi is not None:
            return lo <= system_score <= hi
        return None  # Cannot determine
    except (ValueError, TypeError):
        return None


def calculate_cohens_kappa(teacher_labels, system_labels):
    """Calculate Cohen's Kappa for agreement between teacher and system."""
    # Map labels to categories
    categories = ["无抑郁", "轻度抑郁", "中度抑郁", "中重度抑郁", "重度抑郁"]

    n = len(teacher_labels)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for t, s in zip(teacher_labels, system_labels):
        t_cat = _map_to_category(t)
        s_cat = _map_to_category(s)
        matrix[t_cat][s_cat] += 1

    # Observed agreement
    po = sum(matrix[cat][cat] for cat in categories) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        row_sum = sum(matrix[cat][c] for c in categories)
        col_sum = sum(matrix[c][cat] for c in categories)
        pe += (row_sum / n) * (col_sum / n)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def _map_to_category(label):
    """Map a severity label to a standard category."""
    if "无抑郁" in label:
        return "无抑郁"
    if "重度抑郁" in label:
        return "重度抑郁"
    if "中重度抑郁" in label:
        return "中重度抑郁"
    if "中度抑郁" in label:
        return "中度抑郁"
    if "轻度抑郁" in label:
        return "轻度抑郁"
    return "未知"


# ============================================================
# Generate comparison report
# ============================================================

def generate_report(summaries, teacher_judgments):
    """Generate a markdown comparison report."""
    report = []
    report.append("# 教师专业判断 vs 系统量表分数 —— 对比分析报告")
    report.append("")
    report.append(f"生成时间：自动生成")
    report.append("")

    # --- Summary statistics ---
    report.append("## 一、总体对比结果")
    report.append("")

    all_comparisons = []
    teacher_phq9_labels = []
    system_phq9_labels = []
    teacher_bdi2_labels = []
    system_bdi2_labels = []

    for case_key, judgment in teacher_judgments.items():
        if case_key not in summaries:
            report.append(f"- [WARN] 案例 `{case_key}` 在系统数据中未找到，已跳过")
            continue

        summary = summaries[case_key]

        # Get T0 scores (pre-treatment baseline) for severity comparison
        phq9_t0 = next((s for s in summary['phq9'] if s['point'] == 'T0'), None)
        bdi2_t0 = next((s for s in summary['bdi2'] if s['point'] == 'T0'), None)

        if not phq9_t0 or not bdi2_t0:
            continue

        teacher_sev = judgment.get('severity_level', '')
        teacher_phq9_lo = judgment.get('phq9_range_lo', None)
        teacher_phq9_hi = judgment.get('phq9_range_hi', None)
        teacher_bdi2_lo = judgment.get('bdi2_range_lo', None)
        teacher_bdi2_hi = judgment.get('bdi2_range_hi', None)

        # PHQ-9 comparison
        phq9_comp = compare_severity_levels(teacher_sev, phq9_t0['score'], 'phq9')
        phq9_comp['score_in_range'] = check_score_in_range(
            teacher_phq9_lo, teacher_phq9_hi, phq9_t0['score']
        )
        phq9_comp['teacher_range'] = (teacher_phq9_lo, teacher_phq9_hi)
        phq9_comp['case'] = case_key

        # BDI-II comparison
        bdi2_comp = compare_severity_levels(teacher_sev, bdi2_t0['score'], 'bdi2')
        bdi2_comp['score_in_range'] = check_score_in_range(
            teacher_bdi2_lo, teacher_bdi2_hi, bdi2_t0['score']
        )
        bdi2_comp['teacher_range'] = (teacher_bdi2_lo, teacher_bdi2_hi)
        bdi2_comp['case'] = case_key

        all_comparisons.append({
            'case': case_key,
            'teacher_severity': teacher_sev,
            'phq9': phq9_comp,
            'bdi2': bdi2_comp,
            'improvement_observed': judgment.get('improvement', ''),
            'suicide_risk': judgment.get('suicide_risk', ''),
        })

        teacher_phq9_labels.append(teacher_sev)
        system_phq9_labels.append(phq9_comp['system_label'])
        teacher_bdi2_labels.append(teacher_sev)
        system_bdi2_labels.append(bdi2_comp['system_label'])

    # --- Metrics ---
    n_cases = len(all_comparisons)
    report.append(f"**对比案例总数：{n_cases}**")
    report.append("")

    if n_cases == 0:
        report.append("[WARN] 没有可对比的案例。请检查 teacher_judgments.json 中的案例编号是否与系统数据匹配。")
        return '\n'.join(report)

    # Exact match rate
    phq9_exact = sum(1 for c in all_comparisons if c['phq9']['exact_match'])
    bdi2_exact = sum(1 for c in all_comparisons if c['bdi2']['exact_match'])

    # Adjacent match rate
    phq9_adj = sum(1 for c in all_comparisons if c['phq9']['adjacent_match'])
    bdi2_adj = sum(1 for c in all_comparisons if c['bdi2']['adjacent_match'])

    # Score range hit rate
    phq9_range_hits = sum(1 for c in all_comparisons
                         if c['phq9']['score_in_range'] is True)
    phq9_range_total = sum(1 for c in all_comparisons
                          if c['phq9']['score_in_range'] is not None)
    bdi2_range_hits = sum(1 for c in all_comparisons
                         if c['bdi2']['score_in_range'] is True)
    bdi2_range_total = sum(1 for c in all_comparisons
                          if c['bdi2']['score_in_range'] is not None)

    # Cohen's Kappa
    phq9_kappa = calculate_cohens_kappa(teacher_phq9_labels, system_phq9_labels)
    bdi2_kappa = calculate_cohens_kappa(teacher_bdi2_labels, system_bdi2_labels)

    report.append("### 1.1 严重等级一致性")
    report.append("")
    report.append("| 指标 | PHQ-9 | BDI-II |")
    report.append("|------|-------|--------|")
    report.append(f"| 精确匹配率 | {phq9_exact}/{n_cases} ({phq9_exact/n_cases*100:.0f}%) | {bdi2_exact}/{n_cases} ({bdi2_exact/n_cases*100:.0f}%) |")
    report.append(f"| 相邻等级匹配率 | {phq9_adj}/{n_cases} ({phq9_adj/n_cases*100:.0f}%) | {bdi2_adj}/{n_cases} ({bdi2_adj/n_cases*100:.0f}%) |")
    report.append(f"| Cohen's Kappa | {phq9_kappa:.3f} | {bdi2_kappa:.3f} |")
    report.append("")

    kappa_interpretation = {
        (0.81, 1.0): "几乎完美一致",
        (0.61, 0.8): "高度一致",
        (0.41, 0.6): "中度一致",
        (0.21, 0.4): "一般一致",
        (0.0, 0.2): "轻微一致",
        (-1.0, 0.0): "低于随机水平",
    }
    for (lo, hi), desc in kappa_interpretation.items():
        if lo <= phq9_kappa <= hi:
            report.append(f"*PHQ-9 Kappa 解读：{desc}*")
            break
    report.append("")

    report.append("### 1.2 分数范围命中率")
    report.append("")
    report.append(f"| 指标 | PHQ-9 | BDI-II |")
    report.append(f"|------|-------|--------|")
    report.append(f"| 分数命中率 | {phq9_range_hits}/{phq9_range_total} ({phq9_range_hits/phq9_range_total*100:.0f}%)" if phq9_range_total > 0 else "| 分数命中率 | N/A |")
    report.append(f"| 分数命中率 | {bdi2_range_hits}/{bdi2_range_total} ({bdi2_range_hits/bdi2_range_total*100:.0f}%)" if bdi2_range_total > 0 else "| 分数命中率 | N/A |")
    report.append("")

    # --- Per-case detail ---
    report.append("## 二、逐案例对比详情")
    report.append("")

    for comp in all_comparisons:
        case = comp['case']
        summary = summaries[case]
        phq9_t0 = next((s for s in summary['phq9'] if s['point'] == 'T0'), None)
        bdi2_t0 = next((s for s in summary['bdi2'] if s['point'] == 'T0'), None)

        report.append(f"### {case}")
        report.append("")
        report.append(f"| 维度 | 老师判断 | 系统量表 | 匹配 |")
        report.append(f"|------|----------|----------|------|")
        report.append(f"| 严重等级 | {comp['teacher_severity']} | PHQ-9: {phq9_t0['severity']} (均分 {phq9_t0['score']:.1f}) | {'[OK] 匹配' if comp['phq9']['exact_match'] else '❌ 不匹配'} |")
        report.append(f"| 严重等级 | {comp['teacher_severity']} | BDI-II: {bdi2_t0['severity']} (均分 {bdi2_t0['score']:.1f}) | {'[OK] 匹配' if comp['bdi2']['exact_match'] else '❌ 不匹配'} |")

        # Score range
        if comp['phq9']['teacher_range'][0] is not None:
            report.append(f"| PHQ-9 估算范围 | {comp['phq9']['teacher_range'][0]}-{comp['phq9']['teacher_range'][1]} | 实际均值 {phq9_t0['score']:.1f} | {'[OK] 命中' if comp['phq9']['score_in_range'] else '❌ 未命中'} |")
        if comp['bdi2']['teacher_range'][0] is not None:
            report.append(f"| BDI-II 估算范围 | {comp['bdi2']['teacher_range'][0]}-{comp['bdi2']['teacher_range'][1]} | 实际均值 {bdi2_t0['score']:.1f} | {'[OK] 命中' if comp['bdi2']['score_in_range'] else '❌ 未命中'} |")

        report.append("")
        report.append(f"- 老师观察到的变化趋势：{comp['improvement_observed']}")
        report.append(f"- 老师判断的自杀风险：{comp['suicide_risk']}")

        # Show score trajectory
        report.append("")
        report.append("**系统量表分数变化轨迹：**")
        report.append("")
        report.append("| 评估点 | PHQ-9 分数 | PHQ-9 程度 | BDI-II 分数 | BDI-II 程度 |")
        report.append("|--------|-----------|-----------|-------------|-------------|")
        for i in range(len(summary['phq9'])):
            p = summary['phq9'][i]
            b = summary['bdi2'][i] if i < len(summary['bdi2']) else None
            b_score = f"{b['score']:.1f}" if b else "N/A"
            b_sev = b['severity'] if b else "N/A"
            report.append(f"| {p['point']} | {p['score']:.1f} | {p['severity']} | {b_score} | {b_sev} |")
        report.append("")

    # --- Improvement trend comparison ---
    report.append("## 三、改善趋势对比")
    report.append("")

    # For each case, check if teacher observed improvement matches score trajectory
    for comp in all_comparisons:
        case = comp['case']
        summary = summaries[case]

        # Calculate score delta from T0 to session_20
        phq9_t0 = next((s['score'] for s in summary['phq9'] if s['point'] == 'T0'), None)
        phq9_s20 = next((s['score'] for s in summary['phq9'] if s['point'] == 'session_20'), None)
        bdi2_t0 = next((s['score'] for s in summary['bdi2'] if s['point'] == 'T0'), None)
        bdi2_s20 = next((s['score'] for s in summary['bdi2'] if s['point'] == 'session_20'), None)

        if phq9_t0 and phq9_s20:
            phq9_delta = phq9_s20 - phq9_t0
            phq9_trend = "改善" if phq9_delta < -5 else "轻度改善" if phq9_delta < -2 else "无明显变化" if phq9_delta < 2 else "恶化"
            report.append(f"- **{case}**: PHQ-9 Δ = {phq9_delta:+.1f}（{phq9_t0:.1f}→{phq9_s20:.1f}, {phq9_trend}）")

        if bdi2_t0 and bdi2_s20:
            bdi2_delta = bdi2_s20 - bdi2_t0
            bdi2_trend = "改善" if bdi2_delta < -10 else "轻度改善" if bdi2_delta < -5 else "无明显变化" if bdi2_delta < 5 else "恶化"
            report.append(f"  BDI-II Δ = {bdi2_delta:+.1f}（{bdi2_t0:.1f}→{bdi2_s20:.1f}, {bdi2_trend}）")

    report.append("")

    # --- Notes ---
    report.append("## 四、解读说明")
    report.append("")
    report.append("1. **精确匹配**：老师的严重等级判断与量表分类完全一致")
    report.append("2. **相邻等级匹配**：老师判断与量表分类相差不超过一个等级（如\"中度\" vs \"中重度\"）")
    report.append("3. **Cohen's Kappa**：排除随机一致后的真实一致性指标。>0.6 为高度一致，>0.8 为几乎完美一致")
    report.append("4. **分数命中**：老师估计的分数范围包含了系统的实际均值分数")
    report.append("5. **系统偏差**：如果老师系统性高估或低估，可能反映出 (a) 模拟对话的症状表现不足/过度，或 (b) 评分标准差异")
    report.append("")
    report.append("---")
    report.append("")
    report.append("[] *此报告由 `_compare_results.py` 自动生成*")

    return '\n'.join(report)


# ============================================================
# Main
# ============================================================

def main():
    os.chdir(BASE_DIR)

    print("=" * 60)
    print("教师判断 vs 系统量表 对比分析工具")
    print("=" * 60)

    # Step 1: Load system data
    print("\n[1/3] 加载系统量表数据...")
    summaries = load_all_summaries()
    print(f"  共加载 {len(summaries)} 组系统评估数据")

    # Step 2: Check for teacher judgments file
    judgments_file = os.path.join(BASE_DIR, 'teacher_judgments.json')

    if not os.path.exists(judgments_file):
        print(f"\n[2/3] 未找到 {judgments_file}，正在生成模板...")
        _generate_template(summaries, judgments_file)
        print(f"\n  [OK] 模板已生成: {judgments_file}")
        print(f"\n  >> 请按以下步骤操作：")
        print(f"     1. 打开 {judgments_file}")
        print(f"     2. 根据老师在 Word 文档中填写的评价表，逐案例填入判断")
        print(f"     3. 保存文件后重新运行: python _compare_results.py")
        return

    # Step 3: Load teacher judgments
    print(f"\n[2/3] 加载老师判断数据...")
    with open(judgments_file, encoding='utf-8') as f:
        teacher_judgments = json.load(f)
    print(f"  共加载 {len(teacher_judgments)} 个案例的老师判断")

    # Step 4: Generate comparison report
    print(f"\n[3/3] 生成对比报告...")
    report = generate_report(summaries, teacher_judgments)

    output_file = os.path.join(BASE_DIR, 'comparison_report.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  [OK] 对比报告已保存: {output_file}")
    print(f"\n{'='*60}")
    print(f"完成！请查看 {output_file}")
    print(f"{'='*60}")


def _generate_template(summaries, output_path):
    """Generate a template JSON file for teacher judgments."""
    # Map to case labels
    variant_to_case = {
        'kbd1': '案例 A',
        'kbd2': '案例 B',  # g1 -> 设置A, g4 -> 设置B
        'kbd3': '案例 C',
        'kbd5': '案例 D',
        'kbd6': '案例 E',
        'kbd7': '案例 F',
    }

    template = {}
    for key, data in summaries.items():
        variant = data['variant']
        group = data['group']
        case_base = variant_to_case.get(variant, variant.upper())

        if group == '4':
            case_label = f"{case_base}-设置B (纯咨询)"
        elif variant == 'kbd2' and group == '1':
            case_label = f"{case_base}-设置A (日常社交)"
        else:
            case_label = case_base

        # Use a clean key
        clean_key = f"{variant}-g{group}"
        template[clean_key] = {
            "_case_label": case_label,
            "_variant": variant,
            "_group": group,
            "severity_level": "请填写：无抑郁 / 轻度抑郁 / 中度抑郁 / 中重度抑郁 / 重度抑郁",
            "phq9_range_lo": "请填写PHQ-9估算最低分(0-27)",
            "phq9_range_hi": "请填写PHQ-9估算最高分(0-27)",
            "bdi2_range_lo": "请填写BDI-II估算最低分(0-63)",
            "bdi2_range_hi": "请填写BDI-II估算最高分(0-63)",
            "improvement": "请填写：明显改善 / 轻度改善 / 无明显变化 / 轻度恶化 / 明显恶化",
            "suicide_risk": "请填写：无明显风险 / 有被动念头 / 有主动念头 / 有具体计划 / 无法判断",
            "symptoms_observed": "请列出观察到的核心症状（如：情绪低落、兴趣减退、疲劳感、无价值感等）",
            "clinical_notes": "其他临床观察或备注（可选）",
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    # Also print summary of what cases exist
    print("\n  系统数据中的案例：")
    for key, data in summaries.items():
        variant = data['variant']
        group = data['group']
        case_base = variant_to_case.get(variant, variant.upper())
        if group == '4':
            label = f"{case_base}-设置B"
        elif variant == 'kbd2' and group == '1':
            label = f"{case_base}-设置A"
        else:
            label = case_base
        print(f"    {key} → {label}")


if __name__ == '__main__':
    main()
