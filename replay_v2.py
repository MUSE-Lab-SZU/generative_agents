"""replay_v2.py — 增强版回放：进度条、循环播放、步进控制"""
import os
import json
from flask import Flask, render_template, request

from replay_protocol import FILE_MOVEMENT, FRAMES_PER_STEP
from simulation_roster import replay_roster

file_movement = FILE_MOVEMENT
frames_per_step = FRAMES_PER_STEP

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
    static_url_path="/static",
)


@app.route("/", methods=["GET"])
def index():
    name = request.args.get("name", "")
    if not name:
        return "请指定回放名称，例如: /?name=sim-test-0507-2330"

    compressed_folder = f"results/compressed/{name}"
    replay_file = f"{compressed_folder}/{file_movement}"
    if not os.path.exists(replay_file):
        return (
            f"找不到回放数据: '{replay_file}'<br />"
            f"请先运行 python compress.py --name {name}"
        )

    with open(replay_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    personas = replay_roster(params)

    # 计算模拟步数（不是帧索引）
    # all_movement 的键包含 "0", "1", ..., "description", "conversation"
    # 取最大的数字键作为最大帧索引，再除以 frames_per_step 得到最大模拟步
    max_frame = 0
    for k in params["all_movement"]:
        if k.isdigit():
            max_frame = max(max_frame, int(k))
    max_sim_step = max_frame // frames_per_step

    return render_template(
        "replay_v2.html",
        persona_names=personas,
        max_sim_step=max_sim_step,
        step=1,
        play_speed=4,
        zoom=0.8,
        frames_per_step=frames_per_step,
        **params,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5052, debug=True, use_reloader=False)
