"""Gradio 多 checkpoint 总体抑郁量表测试页面。"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gradio as gr

from customization.checkpoint_depression_scale.assessment import (
    checkpoints_for_agent, list_agents, list_archives, run_batch,
)

HEADERS = ["Checkpoint", "模拟时间", "第1题", "第2题", "第3题", "第4题", "第5题", "总分 / 20", "状态"]
STATUS = {"completed": "完成", "incomplete": "回答无法计分", "error": "测试失败"}


def on_archive_change(archive):
    agents = list_agents(archive) if archive else []
    return gr.update(choices=agents, value=agents[0] if agents else None), gr.update(choices=[], value=[])


def on_agent_change(archive, agent):
    choices = checkpoints_for_agent(archive, agent) if archive and agent else []
    return gr.update(choices=choices, value=[])


def run_assessments(archive, agent, checkpoints, progress=gr.Progress()):
    rows, records = [], []
    yield rows, records, "正在准备测试…", None

    def report(position, count, checkpoint, question_id, attempt):
        fraction = ((position - 1) + max(0, question_id - 1) / 5) / count
        detail = f"第 {question_id}/5 题（第 {attempt} 次回答）" if question_id else "加载中"
        progress(fraction, desc=f"{position}/{count} · {checkpoint} · {detail}")

    output = None
    try:
        for record, output in run_batch(archive, agent, checkpoints, progress=report):
            records.append(record)
            simulation_time = record["simulation_time"]
            if isinstance(simulation_time, dict):
                simulation_time = simulation_time.get("start", "")
            scores = [record["item_scores"].get(str(n)) for n in range(1, 6)]
            rows.append([record["checkpoint"], simulation_time, *scores,
                         record["total_score"], STATUS[record["status"]]])
            yield rows, records, f"已记录 {len(records)}/{len(set(checkpoints))} 个存档点。结果：{output}", str(output)
    except Exception as exc:
        yield rows, records, f"测试中止：{exc}", str(output) if output else None
        return
    completed = sum(record["status"] == "completed" for record in records)
    progress(1, desc="测试结束")
    yield rows, records, f"测试结束：{completed}/{len(records)} 个存档点完成计分。结果：{output}", str(output)


def build_ui():
    archives = list_archives()
    archive = archives[0] if archives else None
    agents = list_agents(archive) if archive else []
    agent = agents[0] if agents else None
    checkpoints = checkpoints_for_agent(archive, agent) if agent else []
    with gr.Blocks(title="Checkpoint 总体抑郁量表测试") as demo:
        gr.Markdown("# 总体抑郁水平及干扰程度量表\n选择模拟存档、agent 和需要测试的存档点。每个存档点独立回答 5 题，总分 0–20。")
        with gr.Row():
            archive_input = gr.Dropdown(choices=archives, value=archive, label="模拟存档")
            refresh = gr.Button("刷新存档")
            agent_input = gr.Dropdown(choices=agents, value=agent, label="Agent")
        checkpoint_input = gr.Dropdown(choices=checkpoints, value=[], multiselect=True, label="待测试存档点（可多选）")
        with gr.Row():
            select_all = gr.Button("选择全部存档点")
            clear_selection = gr.Button("清空选择")
            start = gr.Button("开始量表测试", variant="primary")
        status = gr.Textbox(label="测试进度与保存位置", interactive=False)
        table = gr.Dataframe(headers=HEADERS, datatype=["str", "str"] + ["number"] * 6 + ["str"], interactive=False, label="分数汇总")
        download = gr.File(label="下载本次 JSONL 结果", interactive=False)
        details = gr.JSON(label="原始回答、计分及失败详情")
        refresh.click(lambda: gr.update(choices=list_archives(), value=None), outputs=archive_input)
        archive_input.change(on_archive_change, archive_input, [agent_input, checkpoint_input])
        agent_input.change(on_agent_change, [archive_input, agent_input], checkpoint_input)
        select_all.click(lambda archive, agent: checkpoints_for_agent(archive, agent) if archive and agent else [],
                         [archive_input, agent_input], checkpoint_input)
        clear_selection.click(lambda: [], outputs=checkpoint_input)
        start.click(run_assessments, [archive_input, agent_input, checkpoint_input],
                    [table, details, status, download], concurrency_limit=1,
                    concurrency_id="checkpoint-scale")
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
