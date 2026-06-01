#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化某存档中某角色的全部记忆（event/thought/chat）。

用法：
1) 直接修改下方“配置区”常量后运行：python visualize_agent_memory.py
2) 或通过命令行覆盖，例如：
   python visualize_agent_memory.py --cp-name sim-test-0513 --agent 卡布达

说明：
- 不依赖项目业务模块，纯标准库读取存档文件；
- 快照文件用于获取 node_id 列表（event/thought/chat）；
- storage/associate/docstore.json 用于回查节点全文和元数据。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"

# =========================
# 配置区（按需修改）
# =========================
CHECKPOINT_DIR = Path("results/checkpoints/sim-test-0525")
SNAPSHOT_FILE = CHECKPOINT_DIR / "simulate-20260527-1530.json"
AGENT_NAME = "卡布达"  # 设为 None 则自动取快照里的第一个角色

# OUTPUT_DIR = CHECKPOINT_DIR / "memory_visualization"
OUTPUT_DIR = "results/experiment_data/sim-test-0525/visualizations"
OUTPUT_NAME_PREFIX = ""  # 留空则自动生成

SORT_BY = "create"  # 可选: create / access / node_id / snapshot
INCLUDE_EMBEDDING_PREVIEW = False
EMBEDDING_PREVIEW_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化单个角色的记忆快照")
    parser.add_argument("--cp-name", default=None, help="实验名，对应 results/checkpoints/<name>")
    parser.add_argument("--checkpoint-dir", default=None, help="直接指定 checkpoint 目录")
    parser.add_argument("--snapshot", default=None, help="快照路径或文件名；未传时自动取最新 simulate-*.json")
    parser.add_argument("--agent", default=None, help="角色名；未传则沿用配置区 AGENT_NAME")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--output-prefix", default=None, help="输出文件名前缀")
    parser.add_argument("--sort-by", choices=["create", "access", "node_id", "snapshot"], default=None, help="输出排序方式")
    parser.add_argument("--include-embedding-preview", action="store_true", help="附带 embedding 预览")
    parser.add_argument("--embedding-preview-dim", type=int, default=None, help="embedding 预览维度")
    return parser.parse_args()


def _latest_snapshot(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {checkpoint_dir}")
    snapshots = sorted(checkpoint_dir.glob("simulate-*.json"))
    if not snapshots:
        raise FileNotFoundError(f"未找到 simulate-*.json: {checkpoint_dir}")
    return snapshots[-1]


def _resolve_checkpoint_dir(raw_cp_name: Optional[str], raw_checkpoint_dir: Optional[str]) -> Path:
    if raw_checkpoint_dir:
        checkpoint_dir = Path(raw_checkpoint_dir)
    elif raw_cp_name:
        checkpoint_dir = CHECKPOINTS_ROOT / raw_cp_name
    else:
        checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = checkpoint_dir.resolve()
    return checkpoint_dir


def _resolve_snapshot_path(raw_snapshot: Optional[str], checkpoint_dir: Path, *, prefer_latest: bool) -> Path:
    if raw_snapshot:
        snapshot_path = Path(raw_snapshot)
        if not snapshot_path.is_absolute():
            checkpoint_candidate = checkpoint_dir / raw_snapshot
            if checkpoint_candidate.exists():
                snapshot_path = checkpoint_candidate
            else:
                snapshot_path = snapshot_path.resolve()
        return snapshot_path
    if prefer_latest:
        return _latest_snapshot(checkpoint_dir)
    snapshot_path = Path(SNAPSHOT_FILE)
    if not snapshot_path.is_absolute():
        snapshot_path = snapshot_path.resolve()
    return snapshot_path


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global CHECKPOINT_DIR
    global SNAPSHOT_FILE
    global AGENT_NAME
    global OUTPUT_DIR
    global OUTPUT_NAME_PREFIX
    global SORT_BY
    global INCLUDE_EMBEDDING_PREVIEW
    global EMBEDDING_PREVIEW_DIM

    prefer_latest = bool(args.cp_name or args.checkpoint_dir)
    checkpoint_dir = _resolve_checkpoint_dir(args.cp_name, args.checkpoint_dir)
    snapshot_path = _resolve_snapshot_path(args.snapshot, checkpoint_dir, prefer_latest=prefer_latest)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif prefer_latest:
        output_dir = checkpoint_dir / "memory_visualization"
    else:
        output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()

    CHECKPOINT_DIR = checkpoint_dir
    SNAPSHOT_FILE = snapshot_path
    if args.agent is not None:
        AGENT_NAME = args.agent or None
    OUTPUT_DIR = output_dir
    if args.output_prefix is not None:
        OUTPUT_NAME_PREFIX = args.output_prefix
    if args.sort_by is not None:
        SORT_BY = args.sort_by
    if args.include_embedding_preview:
        INCLUDE_EMBEDDING_PREVIEW = True
    if args.embedding_preview_dim is not None:
        EMBEDDING_PREVIEW_DIM = max(1, args.embedding_preview_dim)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    errors = []
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception as exc:  # pragma: no cover
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"JSON 解析失败: {path}\n" + "\n".join(errors))


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "output"


def _format_ts(raw: Any) -> str:
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        dt = datetime.strptime(raw, "%Y%m%d-%H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw


def _extract_bucket_refs(agent_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    mem = ((agent_payload or {}).get("associate") or {}).get("memory") or {}
    out = {"event": [], "thought": [], "chat": []}
    for bucket in out:
        values = mem.get(bucket, [])
        if isinstance(values, list):
            out[bucket] = [v for v in values if isinstance(v, str)]
    return out


def _extract_doc_nodes(docstore_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = docstore_json.get("docstore/data", {})
    if not isinstance(raw, dict):
        return {}
    data: Dict[str, Dict[str, Any]] = {}
    for node_id, wrapper in raw.items():
        node = {}
        if isinstance(wrapper, dict):
            node = wrapper.get("__data__", {})
            if isinstance(node, str):
                try:
                    node = json.loads(node)
                except Exception:
                    node = {}
        data[str(node_id)] = node if isinstance(node, dict) else {}
    return data


def _extract_vector_embeddings(vector_store_json: Dict[str, Any]) -> Dict[str, List[float]]:
    emb = vector_store_json.get("embedding_dict", {})
    if not isinstance(emb, dict):
        return {}
    out: Dict[str, List[float]] = {}
    for node_id, vec in emb.items():
        if isinstance(vec, list):
            out[str(node_id)] = vec
    return out


def _extract_vector_metadata(vector_store_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata_dict = vector_store_json.get("metadata_dict", {})
    if not isinstance(metadata_dict, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for node_id, metadata in metadata_dict.items():
        if isinstance(metadata, dict):
            out[str(node_id)] = metadata
    return out


def _build_ordered_refs(bucket_refs: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    ordered: List[str] = []
    node_to_buckets: Dict[str, List[str]] = {}
    node_to_bucket_pos: Dict[str, Dict[str, int]] = {}
    for bucket in ("event", "thought", "chat"):
        for idx, node_id in enumerate(bucket_refs.get(bucket, [])):
            if node_id not in node_to_buckets:
                ordered.append(node_id)
                node_to_buckets[node_id] = []
                node_to_bucket_pos[node_id] = {}
            node_to_buckets[node_id].append(bucket)
            if bucket not in node_to_bucket_pos[node_id]:
                node_to_bucket_pos[node_id][bucket] = idx
    return ordered, node_to_buckets, node_to_bucket_pos


def _build_rows(
    ordered_ids: List[str],
    node_to_buckets: Dict[str, List[str]],
    node_to_bucket_pos: Dict[str, Dict[str, int]],
    doc_nodes: Dict[str, Dict[str, Any]],
    vector_embeddings: Dict[str, List[float]],
    vector_metadata: Dict[str, Dict[str, Any]],
    agent_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot_idx, node_id in enumerate(ordered_ids):
        node = doc_nodes.get(node_id, {})
        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        v_metadata = vector_metadata.get(node_id, {})
        access = v_metadata.get("access", metadata.get("access", ""))
        text = node.get("text", "")
        embedding = node.get("embedding")
        if not isinstance(embedding, list):
            embedding = vector_embeddings.get(node_id, [])
        embedding_dim = len(embedding) if isinstance(embedding, list) else 0

        row = {
            "agent_name": agent_name,
            "snapshot_order": snapshot_idx,
            "node_id": node_id,
            "memory_bucket": "|".join(node_to_buckets.get(node_id, [])),
            "memory_bucket_pos": ";".join(
                f"{k}:{v}" for k, v in node_to_bucket_pos.get(node_id, {}).items()
            ),
            "node_type": metadata.get("node_type", ""),
            "text": text if isinstance(text, str) else str(text),
            "subject": metadata.get("subject", ""),
            "predicate": metadata.get("predicate", ""),
            "object": metadata.get("object", ""),
            "address": metadata.get("address", ""),
            "poignancy": metadata.get("poignancy", ""),
            "create": metadata.get("create", ""),
            "expire": metadata.get("expire", ""),
            "access": access,
            "create_fmt": _format_ts(metadata.get("create", "")),
            "expire_fmt": _format_ts(metadata.get("expire", "")),
            "access_fmt": _format_ts(access),
            "embedding_dim": embedding_dim,
            "node_found": bool(node),
        }

        if INCLUDE_EMBEDDING_PREVIEW and embedding_dim > 0:
            row["embedding_preview"] = embedding[: max(1, EMBEDDING_PREVIEW_DIM)]
        else:
            row["embedding_preview"] = []
        rows.append(row)
    return rows


def _sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    if sort_by == "snapshot":
        return sorted(rows, key=lambda r: (r.get("snapshot_order", 0), r.get("node_id", "")))
    if sort_by == "node_id":
        return sorted(rows, key=lambda r: r.get("node_id", ""))
    if sort_by in {"create", "access"}:
        return sorted(
            rows,
            key=lambda r: (
                r.get(sort_by, ""),
                r.get("snapshot_order", 0),
                r.get("node_id", ""),
            ),
        )
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "agent_name",
        "snapshot_order",
        "node_id",
        "memory_bucket",
        "memory_bucket_pos",
        "node_type",
        "poignancy",
        "create",
        "create_fmt",
        "expire",
        "expire_fmt",
        "access",
        "access_fmt",
        "subject",
        "predicate",
        "object",
        "address",
        "embedding_dim",
        "node_found",
        "text",
        "embedding_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["embedding_preview"] = json.dumps(out.get("embedding_preview", []), ensure_ascii=False)
            writer.writerow(out)


def _html_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_html(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    title = f"角色记忆可视化 - {meta.get('agent_name', '')}"
    row_html: List[str] = []
    for idx, r in enumerate(rows, start=1):
        keyword_blob = " ".join(
            [
                str(r.get("node_id", "")),
                str(r.get("memory_bucket", "")),
                str(r.get("node_type", "")),
                str(r.get("subject", "")),
                str(r.get("predicate", "")),
                str(r.get("object", "")),
                str(r.get("address", "")),
                str(r.get("text", "")),
            ]
        ).lower()
        text_cell = _html_escape(r.get("text", ""))
        preview = _html_escape(r.get("embedding_preview", []))
        row_html.append(
            "<tr "
            f"data-keywords=\"{_html_escape(keyword_blob)}\" "
            f"data-bucket=\"{_html_escape(r.get('memory_bucket', ''))}\" "
            f"data-nodetype=\"{_html_escape(r.get('node_type', ''))}\" "
            f"data-found=\"{str(r.get('node_found', False)).lower()}\""
            ">"
            f"<td>{idx}</td>"
            f"<td>{_html_escape(r.get('node_id', ''))}</td>"
            f"<td>{_html_escape(r.get('memory_bucket', ''))}</td>"
            f"<td>{_html_escape(r.get('node_type', ''))}</td>"
            f"<td>{_html_escape(r.get('poignancy', ''))}</td>"
            f"<td>{_html_escape(r.get('create_fmt', r.get('create', '')))}</td>"
            f"<td>{_html_escape(r.get('expire_fmt', r.get('expire', '')))}</td>"
            f"<td>{_html_escape(r.get('access_fmt', r.get('access', '')))}</td>"
            f"<td>{_html_escape(r.get('subject', ''))}</td>"
            f"<td>{_html_escape(r.get('predicate', ''))}</td>"
            f"<td>{_html_escape(r.get('object', ''))}</td>"
            f"<td>{_html_escape(r.get('address', ''))}</td>"
            f"<td>{_html_escape(r.get('embedding_dim', ''))}</td>"
            f"<td>{_html_escape(r.get('node_found', ''))}</td>"
            "<td><details><summary>展开</summary>"
            f"<div class=\"text-cell\">{text_cell}</div>"
            "</details></td>"
            "<td><details><summary>展开</summary>"
            f"<div class=\"mono\">{preview}</div>"
            "</details></td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --fg: #1e2530;
      --card: #ffffff;
      --line: #e0e5ec;
      --blue: #1f6feb;
      --warn: #b42318;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 12px;
    }}
    .title {{
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .meta {{
      line-height: 1.8;
      word-break: break-all;
    }}
    .filters {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .filters input, .filters select {{
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 160px;
    }}
    .count {{
      margin-left: auto;
      color: var(--blue);
      font-weight: 600;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 1450px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f2f5f9;
      z-index: 1;
    }}
    .text-cell {{
      white-space: pre-wrap;
      max-width: 560px;
      line-height: 1.5;
    }}
    .mono {{
      font-family: Consolas, "Courier New", monospace;
      white-space: pre-wrap;
      max-width: 560px;
    }}
    .warn {{
      color: var(--warn);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">{_html_escape(title)}</div>
      <div class="meta">
        checkpoint: {_html_escape(meta.get("checkpoint_dir", ""))}<br/>
        snapshot: {_html_escape(meta.get("snapshot_file", ""))}<br/>
        agent: {_html_escape(meta.get("agent_name", ""))}<br/>
        total_rows: {_html_escape(meta.get("total_rows", 0))}<br/>
        missing_nodes: <span class="{ 'warn' if meta.get('missing_nodes', 0) else '' }">{_html_escape(meta.get("missing_nodes", 0))}</span>
      </div>
    </div>

    <div class="card filters">
      <input id="kw" placeholder="关键词检索（node/text/subject/object...）" />
      <select id="bucket">
        <option value="">全部 bucket</option>
        <option value="event">event</option>
        <option value="thought">thought</option>
        <option value="chat">chat</option>
      </select>
      <select id="nodeType">
        <option value="">全部 node_type</option>
        <option value="event">event</option>
        <option value="thought">thought</option>
        <option value="chat">chat</option>
      </select>
      <label><input type="checkbox" id="onlyMissing" /> 仅看缺失节点</label>
      <div class="count" id="countLabel"></div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>node_id</th>
            <th>memory_bucket</th>
            <th>node_type</th>
            <th>poignancy</th>
            <th>create</th>
            <th>expire</th>
            <th>access</th>
            <th>subject</th>
            <th>predicate</th>
            <th>object</th>
            <th>address</th>
            <th>embedding_dim</th>
            <th>node_found</th>
            <th>text</th>
            <th>embedding_preview</th>
          </tr>
        </thead>
        <tbody id="tbody">
          {"".join(row_html)}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const kw = document.getElementById("kw");
    const bucket = document.getElementById("bucket");
    const nodeType = document.getElementById("nodeType");
    const onlyMissing = document.getElementById("onlyMissing");
    const countLabel = document.getElementById("countLabel");
    const rows = Array.from(document.querySelectorAll("#tbody tr"));

    function applyFilter() {{
      const q = (kw.value || "").trim().toLowerCase();
      const b = (bucket.value || "").trim().toLowerCase();
      const nt = (nodeType.value || "").trim().toLowerCase();
      const miss = !!onlyMissing.checked;
      let shown = 0;
      for (const tr of rows) {{
        const words = (tr.dataset.keywords || "");
        const rb = (tr.dataset.bucket || "").toLowerCase();
        const rnt = (tr.dataset.nodetype || "").toLowerCase();
        const rf = (tr.dataset.found || "false") === "true";
        const okQ = !q || words.includes(q);
        const okB = !b || rb.includes(b);
        const okT = !nt || rnt === nt;
        const okM = !miss || !rf;
        const show = okQ && okB && okT && okM;
        tr.style.display = show ? "" : "none";
        if (show) shown += 1;
      }}
      countLabel.textContent = "显示 " + shown + " / " + rows.length;
    }}

    kw.addEventListener("input", applyFilter);
    bucket.addEventListener("change", applyFilter);
    nodeType.addEventListener("change", applyFilter);
    onlyMissing.addEventListener("change", applyFilter);
    applyFilter();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    _apply_cli_overrides(args)

    checkpoint_dir = Path(CHECKPOINT_DIR)
    snapshot_path = Path(SNAPSHOT_FILE)
    if not snapshot_path.is_absolute():
        snapshot_path = snapshot_path.resolve()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = checkpoint_dir.resolve()

    snapshot = _load_json(snapshot_path)
    agents = snapshot.get("agents", {})
    if not isinstance(agents, dict) or not agents:
        raise ValueError("快照中不存在 agents 或结构异常。")

    agent_name = AGENT_NAME
    if not agent_name:
        agent_name = next(iter(agents.keys()))
        print(f"[INFO] AGENT_NAME 未配置，自动使用: {agent_name}")
    if agent_name not in agents:
        available = ", ".join(agents.keys())
        raise ValueError(f"找不到角色: {agent_name}\n可选角色: {available}")

    bucket_refs = _extract_bucket_refs(agents[agent_name])
    ordered_ids, node_to_buckets, node_to_bucket_pos = _build_ordered_refs(bucket_refs)
    if not ordered_ids:
        raise ValueError(f"角色 {agent_name} 在快照中未找到 associate.memory 引用。")

    associate_dir = checkpoint_dir / "storage" / agent_name / "associate"
    docstore_path = associate_dir / "docstore.json"
    vector_store_path = associate_dir / "default__vector_store.json"

    docstore_json = _load_json(docstore_path)
    doc_nodes = _extract_doc_nodes(docstore_json)

    vector_embeddings: Dict[str, List[float]] = {}
    vector_metadata: Dict[str, Dict[str, Any]] = {}
    if vector_store_path.exists():
        try:
            vector_store_json = _load_json(vector_store_path)
            vector_embeddings = _extract_vector_embeddings(vector_store_json)
            vector_metadata = _extract_vector_metadata(vector_store_json)
        except Exception:
            vector_embeddings = {}
            vector_metadata = {}

    rows = _build_rows(
        ordered_ids=ordered_ids,
        node_to_buckets=node_to_buckets,
        node_to_bucket_pos=node_to_bucket_pos,
        doc_nodes=doc_nodes,
        vector_embeddings=vector_embeddings,
        vector_metadata=vector_metadata,
        agent_name=agent_name,
    )
    rows = _sort_rows(rows, SORT_BY)

    missing_nodes = sum(1 for r in rows if not r.get("node_found", False))
    total_rows = len(rows)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_prefix = f"memory_view_{agent_name}_{snapshot_path.stem}"
    prefix = _safe_filename(OUTPUT_NAME_PREFIX.strip() if OUTPUT_NAME_PREFIX else default_prefix)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    html_path = output_dir / f"{prefix}.html"

    payload = {
        "meta": {
            "checkpoint_dir": str(checkpoint_dir),
            "snapshot_file": str(snapshot_path),
            "agent_name": agent_name,
            "total_rows": total_rows,
            "missing_nodes": missing_nodes,
            "sort_by": SORT_BY,
        },
        "bucket_counts": {k: len(v) for k, v in bucket_refs.items()},
        "rows": rows,
    }

    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    html_str = _build_html(payload["meta"], rows)
    html_path.write_text(html_str, encoding="utf-8")

    print("[OK] 记忆可视化文件已生成")
    print(f"[OUT] JSON: {json_path}")
    print(f"[OUT] CSV : {csv_path}")
    print(f"[OUT] HTML: {html_path}")
    print(f"[INFO] total_rows={total_rows}, missing_nodes={missing_nodes}")


if __name__ == "__main__":
    main()