# 运行脚本目录

| 位置 | 用途 |
| --- | --- |
| 根层 `run_batch_then_repeat_eval.sh`、`run_resume_batch_then_repeat_eval.sh` | 当前批量实验与恢复入口 |
| 根层 `*.py` | 批量/单次实验编排、量表 worker、评估与恢复工具；相互 import 的 Python 文件暂留同层 |
| `services/vllm/`、`services/ollama/`、`services/network/` | 模型服务、下载、安装、连通性检查和代理 |
| `simulation/` | 单次仿真启动、批跑、状态查看与回放 |
| `humanlike_validation/` | 仿真后类人验证的 shell 配置与独立离线入口 |
| `maintenance/` | 项目与结果同步脚本 |
| `legacy/` | 历史 G1/G2 实验 shell 入口 |

常用命令从仓库根目录执行，例如 `bash runshells/services/vllm/vllm_services.sh start`、`bash runshells/simulation/sim_status.sh <名称>`、`bash runshells/humanlike_validation/run_psi_bench_eval.sh <存档目录>`。原有类人验证 Python 入口仍在 `runshells/` 根层；独立的长期病例一致性入口在 `runshells/humanlike_validation/run_longitudinal_case_fidelity.py`，0929 全组入口在 `runshells/humanlike_validation/run_longitudinal_0929_batch.py`，源文本复核入口在 `runshells/humanlike_validation/apply_0929_source_audit.py`，分析实现在 `humanlike_validation/`。新增 shell 脚本按用途进入对应子目录，并按脚本自身所在深度定位项目根目录（普通子目录为 `../..`，`services/*/` 为 `../../..`）；跨脚本调用使用仓库根目录相对路径。新增 Python 文件若与现有编排模块互相 import，先确认 import 和 `__file__` 路径假设，再决定是否移入子目录。
