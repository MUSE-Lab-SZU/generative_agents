# NMH / Agentopia 论文 LaTeX 目录

本目录包含两条投稿线的 LaTeX 源文件：NMH（Nature Mental Health 主投）与 Agentopia 风格备选版。

## 文件结构

```
writing/latex/
├── README.md                          ← 本文件
├── nmh-manuscript.tex                 ← NMH 英文投稿版（2026-10-08 校准至代码 3e8259d）
├── nmh-manuscript-zh.tex              ← NMH 中文对照审读版（不用于投稿）
├── nmh-references.bib                 ← NMH 参考文献（英/中两版共用）
├── agentopia-style-manuscript.tex     ← Agentopia 风格英文版
├── agentopia-style-manuscript-zh.tex  ← Agentopia 风格中文镜像版
├── agentopia-references.bib           ← Agentopia 参考文献
├── sn-article-template/               ← Springer Nature 模板（sn-jnl.cls、sn-nature.bst 等）
│   └── （仅模板文件；nmh 主文件已提到根目录，经 TEXINPUTS 引用）
├── backup/                            ← 历史版本归档（v2 tex/bib/pdf）
└── overleaf-upload/                   ← Overleaf 上传打包
```

版本沿革：v2（2026-08-12）→ v3（2026-09-27 重写）→ 现 nmh-manuscript.tex（2026-10-08 去版本号转正）。
文件名不再带版本号；历史版本只存在于 `backup/` 与 git 历史。

## 编译方法

### NMH 英文版（根目录，模板在 sn-article-template/）

```bash
cd writing/latex
TEXINPUTS=".;sn-article-template;" BSTINPUTS=".;sn-article-template;" \
  latexmk -pdf nmh-manuscript.tex
```

### NMH 中文审读版

```bash
cd writing/latex
latexmk -xelatex nmh-manuscript-zh.tex
```

### Agentopia 双语版

```bash
cd writing/latex
latexmk -pdf agentopia-style-manuscript.tex          # 英文
latexmk -xelatex agentopia-style-manuscript-zh.tex   # 中文
```

### Overleaf

将 `sn-article-template/` 整目录 + 根目录的 `nmh-manuscript.tex`、`nmh-references.bib` 一起上传，主文件设为 `nmh-manuscript.tex`。

## 占位符说明（NMH 版）

论文中以下占位符待目标批次（0929，LRN 八组×5）实验核验后填入：

| 占位符 | 含义 |
|--------|------|
| `[ModelName]` | 平台名称，待命名（术语表锁定项） |
| `[X.X]` / `[X.XX]` / `[XX%]` | 统计数值（均值、效应量、ICC、百分比等） |
| `[N]` | 样本量（独立 run 数等） |
| `[Author N]` / `[姓名]` / `[资助]` | 作者与致谢信息 |
| `[Evidence needed: ...]` | 缺失证据说明（对应 claim-evidence map） |

## 关键约束（来自交接文档，勿违背）

1. **C-mech 因果降级**: 用 `consistent with a within-simulation causal role`，非 `demonstrates causally`
2. **两量表**: 主线 PHQ-9 + BDI-II，SDS 未启用不入正文
3. **T4 双条件**: 停止干预后独立 post-simulation 期间 + 不施测量表 + 冻结 checkpoint 恢复
4. **三层统计单位**: N = 独立 run，复评不计入 N
5. **controller 分批不混池**: legacy / progressive-d 分开分析
6. **动词校准**: 2.1+2.3 用 show/produced；2.4 用 consistent with；避免 first/novel/demonstrates causally
