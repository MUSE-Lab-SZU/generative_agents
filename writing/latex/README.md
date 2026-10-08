# NMH 论文 LaTeX 目录

本目录包含 Nature Mental Health 稿件的 LaTeX 源文件及 Springer Nature 官方模板。

## 文件结构

```
writing/latex/
├── README.md                          ← 本文件
├── nmh-manuscript.tex                 ← 论文主文件（根级副本，方便编辑）
├── nmh-references.bib                 ← 参考文献数据库（根级副本）
└── sn-article-template/               ← 可编译目录（含类文件 + bst）
    ├── nmh-manuscript.tex             ← 论文主文件（编译用，与根级同步）
    ├── nmh-references.bib             ← 参考文献数据库（编译用，与根级同步）
    ├── sn-jnl.cls                     ← Springer Nature 文档类（v2.1, 2023-04）
    ├── sn-nature.bst                  ← Nature Portfolio 参考文献格式
    ├── sn-article.tex                 ← 模板示例（参考，勿删）
    ├── sn-bibliography.bib            ← 模板示例 bib（参考）
    ├── sn-basic.bst / sn-vancouver.bst / ...  ← 其他参考格式（备用）
    ├── user-manual.pdf                ← 模板使用手册
    └── sn-article.pdf                 ← 模板编译示例
```

## 论文信息

- **标题**: In silico separation of cognitive restructuring and memory consolidation in simulated depression treatment with generative agents
- **定位**: B — 发现/机制论文（双机制分离框架）
- **来源草稿**: `writing/2026-08-12-NMH_manuscript_v2_重构B双机制分离.md`
- **模板**: Springer Nature LaTeX (`sn-jnl.cls`)，参考风格 `sn-nature`（Nature Portfolio 编号制）
- **Skill**: nature-writing (research / full / zh-to-en / nature-family)

## 编译方法

### 方式一：命令行（推荐）

```bash
cd writing/latex/sn-article-template
latexmk -pdf nmh-manuscript.tex
```

或手动三步：

```bash
cd writing/latex/sn-article-template
pdflatex nmh-manuscript
bibtex   nmh-manuscript
pdflatex nmh-manuscript
pdflatex nmh-manuscript
```

输出：`nmh-manuscript.pdf`（约 11 页）

### 方式二：Overleaf

将 `sn-article-template/` 目录整体上传为 Overleaf 项目，主文件设为 `nmh-manuscript.tex`。

### 方式三：VS Code + LaTeX Workshop

打开 `sn-article-template/` 文件夹，编译 `nmh-manuscript.tex`。

## 占位符说明

论文中以下占位符待目标批次实验核验后填入：

| 占位符 | 含义 | 来源 |
|--------|------|------|
| `[ModelName]` | 平台名称，待命名 | 术语表锁定项 |
| `[X.X]` / `[XX%]` | 统计数值（均值、效应量、百分比等） | 待核验目标批次 |
| `[N]` | 样本量（人设数、独立 run 数） | 待核验目标批次 |
| `[X.XX]` | Cohen's d / ICC / κ 等系数 | 待核验目标批次 |
| `[Author N]` | 作者信息 | 待填 |
| `[Evidence needed: ...]` | 缺失证据说明 | v2 claim-evidence map |

## 关键约束（来自交接文档，勿违背）

1. **C-mech 因果降级**: 用 `consistent with a within-simulation causal role`，非 `demonstrates causally`
2. **两量表**: 主线 PHQ-9 + BDI-II，SDS 未启用不入正文
3. **T4 双条件**: 停止干预后独立 post-simulation 期间 + 不施测量表 + 冻结 checkpoint 恢复
4. **三层统计单位**: N = 独立 run，复评不计入 N
5. **controller 分批不混池**: legacy / progressive-d 分开分析
6. **动词校准**: 2.1+2.3 用 show/produced；2.4 用 consistent with；避免 first/novel/demonstrates causally
