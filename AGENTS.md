# AGENTS.md - Codebase Guide for Agentic Development

## Project Overview

A Chinese-language technical blog built with **Quartz 4** and published to GitHub Pages. `content/` holds plain Markdown notes; the build renders them into `public/`. `quartz/` is vendored Quartz with a handful of local components, and `README.md` documents the content model in detail.

### Technology Stack
- **Build**: Quartz 4.0 (`npm run build` → `quartz` CLI + `scripts/create-clean-url-pages.mjs`)
- **Languages**: TypeScript (build/components), SCSS, Markdown
- **Content**: Markdown + `content/article-index.json` for titles, dates and tags
- **Markdown pipeline**: remark/rehype — GFM, Obsidian-flavored Markdown, `rehype-pretty-code`, `rehype-katex`
- **Runtime extras**: in-browser Pyodide runner for `python` code blocks, KaTeX, client-side search
- **Deployment**: GitHub Pages via `.github/workflows/deploy.yml` on push to `main`

---

## Build, Lint, Test Commands

```bash
npm ci                 # install
npm run serve          # build + local preview
npm run build          # write public/
npm run check          # tsc --noEmit + prettier --check
npm run format         # prettier --write
npm test               # tsx quartz/util/path.test.ts && tsx quartz/depgraph.test.ts
npm run ci             # check + build
```

- `npm run build` regenerates `public/` (gitignored). Never edit `public/` by hand.
- `npm run check` runs Prettier over the repo. `.prettierignore` excludes `content/**/*.md` (Prettier reinterprets LaTeX underscores as emphasis) and `quartz/static/katex/`, so article prose is never reformatted — keep paragraphs on one line yourself.
- Tests are Quartz's own TS tests; there is no test suite for article content. Article correctness is established by re-running the embedded Python snippets (see below).

---

## Content Model

```
content/
├── article-index.json      # titles, dates, tags, series descriptions
├── index.md                # 首页：个人介绍 + 最近文章（autoRecent: true）
├── articles.md             # 文章索引页（articleIndex: true）
├── learning/
│   ├── index.md            # section landing page (has frontmatter)
│   ├── assets/             # every image, referenced as /learning/assets/<name>
│   ├── *.md                # standalone notes (camelCase.md)
│   └── <series>/           # e.g. flash-attention/, precision/
│       └── NN-kebab-name.md
└── pitfalls/
```

- **首页就是 `content/index.md`**：个人介绍写在正文里，最近文章由 `autoRecent: true` 触发的 `HomeRecentNotes` 渲染在正文之后（`afterBody`）。所以没有单独的「关于」页——介绍和文章列表在同一页上。

- **Articles carry no frontmatter.** Title, `date` and `tags` live in `content/article-index.json`, keyed by slug (path relative to `content/`, no `.md`). A missing entry means the article gets no title and no tags.
- **A series is a folder.** Files named `NN-` order the series; the folder path is the series slug. `article-index.json` `series` entries only override the title/description. Do **not** add `series`/`seriesOrder` fields to article records — the build errors.
- **File naming**: standalone notes `camelCase.md` (`TurboQuant.md`, `roofline.md`); series articles `NN-kebab-case.md` (`01-flash-attention.md`, `16-block-sparse.md`); config/component files `kebab-case.ts`.
- **Sections**: a new section is a directory under `content/` plus an `index.md` with `title`/`date` frontmatter.
- **One paragraph is one long line.** Do not hard-wrap Chinese prose.
- **Code fences always declare a language** (`python`, `cpp`, `text`, `bash`, `ptx`, `pseudocode`). `text` is used for program output.

### 伪代码用 pseudocode 围栏，不要用 text

论文里的算法（Algorithm 1、Algorithm 2 这类）排版成算法列表：上下两条粗线、标题行、左侧行号栏、关键字加粗、注释右置。这是 LaTeX `algorithm2e` 的样子，也是读者对"论文算法"的预期版式。渲染器在 `quartz/plugins/transformers/pseudocode.ts`，样式在 `quartz/styles/custom.scss` 的 `.algo*`。

````
```pseudocode title="Algorithm 1: FlashAttention 前向"
for j in 1..T_c:
    for i in 1..T_r:
        S_ij = Q_i K_j^T        # 在片上算
```
````

- **行号自动生成**，不要在正文里手写 `1.` `2.`；手写的行首编号会被剥掉。
- 缩进取块内最小非零缩进量，2 空格 / 4 空格都行。
- **行内注释**写作同行尾随的 `▷` / `//` / `#`（前面要有空白），会被移到右侧斜体显示；跨行铺开的注释做不到。**不要用 `←` 写注释**——它在算法记法里是赋值号（`O_i ← diag(ℓ)^{-1} O_i`），渲染器不会把它当注释。
- `title=` 用论文里的算法名；不是照搬某篇论文的，用一句描述。
- 块内的数学记号（`S_ij`、`K_j^T`）**保持平文本**——这里不是 KaTeX 渲染区，和自绘 SVG 同理。
- **不要**把程序输出、ASCII 图、代码结构说明也改成 `pseudocode`；那三类继续用 `text`。

### Adding an article
1. Create the `.md` under the right section (or series folder with an `NN-` prefix).
2. Add an `articles` entry keyed by slug with `title`, `date` (`YYYY-MM-DD`) and at least one tag.
3. Add the tag list consistently with siblings — `/tags`, search and RSS pick it up automatically.
4. `npm run build`, then check `public/<slug>.html`.

---

## Runnable Python Snippets

`quartz/components/scripts/runCode.inline.ts` turns every `python` block into a **▶ 运行** button. Blocks execute in **Pyodide 0.26.4** (browser), which ships **NumPy 1.26.4** and **mpmath**; there is **no torch, no CUDA, no transformer_engine**. All blocks in one page share one interpreter and persist their state, so a later block may use names defined earlier on the same page — the same is true of your reproduction environment.

- Any block whose output the article prints must reproduce **byte-for-byte** under NumPy 1.26.4. Verify with a NumPy 1.26.4 interpreter, not a newer one; `np.trapezoid` (NumPy 2.0+) is a recurring trap and the wrapper here has been wrong before.
- Label printed output with the environment that produced it when it was not the browser runner, e.g. `真实输出（Python 3.12；与站点内置运行器的 Pyodide 0.26.4 + NumPy 1.26.4 一致）`.
- A snippet that genuinely needs a GPU or torch must say so in the surrounding prose. Never fabricate an output block for one.
- `cpp`/`ptx`/CuTe-DSL fences are excerpts for reading, not runnable demos.

---

## Math Rendering (KaTeX)

Quartz `Plugin.Latex` with `renderEngine: "katex"`; the CSS and fonts are **self-hosted** at `/static/katex/` (see `quartz/plugins/transformers/latex.ts`), not loaded from a CDN.

- **Inline math** — single dollar signs: `$...$`
- **Display math** — double dollar signs on their own lines: `$$...$$`
- **NEVER wrap math in backticks.** `` `$...$` `` renders as literal code (the `$` and `\command` visible) and KaTeX never runs. This was a site-wide bug in the flash-attention series (637 spans). Write `$...$` directly in prose, table cells and list items.
- Keep `$` delimiters balanced on the same line; do not put `$...$` inside a code span or a backtick-wrapped title (e.g. a paper title like `Self-attention Does Not Need $O(n^2)$ Memory`).
- If a built page shows raw `$...$` / `\command` text, look for stray backticks around the formula first — they swallow the `$$...$$` block too.
- Unbalanced `$` in a **table row** breaks the row; escape `\|` inside `[[wikilinks]]` used in tables.

### 自绘 SVG 里的数学（KaTeX 管不到）

KaTeX 只处理 Markdown，**不会进入 SVG**。所以 `content/learning/assets/*.svg` 里写 `$...$` 只会原样显示美元符号，写 `S_ij` / `K_j^T` / `exp(` 就是平文本——看起来像"公式没渲染"。

- SVG 经 `<img src="…svg">` 加载，因此 `<foreignObject>`、`<script>`、外部 CSS **都不执行**，别指望它们。
- 下标/上标只能用 `<tspan>` 的 `dy` + `font-size` 做位移：

  | 级别 | dy（相对上一片段） | font-size |
  | --- | --- | --- |
  | 基线 | 0 | 15 |
  | 上标 | −7 | 10.5 |
  | 下标 | +4 | 10.5 |
  | 上标里的下标 | +2 | 8 |

- **`dy` 是累积的**，每个偏移必须成对归还，否则后续文字整体漂移。改完打印 dy 序列，确认最终回到 0。
- **只给数学记号排版**：`S_ij`、`K_j^T`、`a_hi`、`S_{t-1}` 是数学，要排成上下标。而 `two_sum`、`Quant_mse`、`idx_j`、`O_buf` 是源码里的函数名/变量名/kernel 名，**平文本才正确**，不要动。
- 数学减号用 U+2212 `−`，不要 ASCII `-`。

---


## Figures（图片与引用）

- **优先使用现有/官方图**（论文、官方文档、官方博客），不要为了美观重画论文或官方的关键概念图。
- **任何从外部来源借用的图片**，必须在其下方标注来源（含 Figure 编号）：
  `> 图源：<来源名>《<标题>》（<链接或出处>）Figure N`
  示例：`> 图源：Dao-AILab《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135）Figure 1`
- **自绘示意图**（自定义 SVG 等）在下方标注：`> 自绘示意图`。
- **转写/重绘的论文图**要经得起对照。写 `按原论文 pgfplots 坐标转写` 就意味着图里每条曲线都必须来自论文源码里的真实坐标，系列名和系列数也要和论文图例一致；只想画其中几条就在正文写明省了哪几条。**不许拿手画的示意数据冒充转写**——这类图看起来最可信，实际最难查。改动前先从论文 e-print（`https://arxiv.org/e-print/<id>`，解包后是 TeX 源码）取出坐标，再据此生成 SVG。
- 借来的图若是**原始图片裁剪**（如论文 Figure 的 PNG 导出），写 `> 图源：…Figure N`；转写的写 `按原论文 <源码文件> 转写`。两者不要混用同一句标注。版式与论文一致但坐标数据是摘要式转写的，写 `> 图源：…（按原论文源码重绘）`，不要冒充原图。
- 图片统一放 `content/learning/assets/`，用站点路径引用（`/learning/assets/...`），不要留孤立图片。
- 图片下方的说明文字遵循 KaTeX 约定，不要用反引号包 `$...$`。

---

## Code Style Guidelines

### TypeScript / TSX (`quartz/`)
- Existing Quartz conventions: Prettier defaults from `.prettierrc`, 2-space indent, no semicolons.
- Components are functions returning JSX; client scripts are `*.inline.ts` and must attach/detach listeners via `window.addCleanup` because the site uses SPA navigation and re-fires `nav` events.
- Never assume a DOM node survives navigation — rebind on every `nav`.

### SCSS (`quartz/styles/`)
- Site overrides go in `custom.scss`; `base.scss`, `variables.scss`, `syntax.scss`, `callouts.scss` are Quartz upstream. Keep local edits in `custom.scss` where possible.
- Class names `kebab-case`; keep dark-mode rules paired with light via the `[saved-theme]` attribute, not `prefers-color-scheme` alone.

### Commits
- [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <subject>`, imperative mood, ≤ 50 chars, no trailing period.
- Types: `feat`, `fix`, `content` (article edits), `docs`, `style`, `refactor`, `perf`, `chore`, `ci`, `test`, `revert`.
- **One type per commit** — split a feature from content edits. Add a body (blank line, wrapped at ~72 chars) when the subject doesn't carry the *what* and *why*.
  - `content: correct CUDA 13 Fedora support facts`
  - `fix: preload mpmath before running mpmath snippets`

---

## Notes for Agents

- Quartz is vendored under `quartz/`; prefer editing `content/`, `quartz/components/`, `quartz/styles/custom.scss` and config. Changes under `quartz/plugins/` and `quartz/util/` are upstream code — touch them only when the task needs it.
- **Most article content is AI-generated and reviewed by hand.** Treat the prose as unverified: expect inconsistent math notation (backtick-wrapped `$...$`, mixed `$...$` vs monospace symbols), formulaic transitions, redundant explanation, and factual/technical drift. Cross-check a claim against the surrounding math and the linked source before trusting it, and fix inconsistencies rather than reproduce them.
- Chinese is the content and UI language; write comments in Chinese.
- Before finishing a content change: `npm run build` succeeds and the affected page renders (math as math, images present, links resolving).
