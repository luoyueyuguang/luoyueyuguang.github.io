# luoyue's Blog

这是一个使用 [Quartz](https://quartz.jzhao.xyz/) 生成的 GitHub Pages 博客。

## 本地预览

```bash
npm ci
npm run serve
```

## GitHub Pages

内容放在 `content/` 目录中。

文章正文保持为纯 Markdown；标题、日期和标签统一维护在
`content/article-index.json`。新增文章时：

1. 在 `content/learning/`、`content/pitfalls/` 等栏目中创建 `.md` 文件，不添加标题
   frontmatter。
2. 在 `content/article-index.json` 的 `articles` 中加入以文章 slug 为键的元数据；slug
   是相对 `content/` 的路径，并省略 `.md`。
3. 为文章设置至少一个标签。构建后首页最近文章、`/articles` 文章索引、搜索、RSS
   和 `/tags` 标签归档都会自动同步。

## 文章系列

系列用于组织具有固定阅读顺序的多篇文章，**系列就是文件夹**：把文章放进一个文件夹，
文件名用 `NN-` 数字前缀标记顺序，该文件夹自动成为系列，无需任何额外配置：

```
content/learning/flash-attention/
├── 01-flash-attention.md
└── 02-flash-attention-2.md
```

- 系列 slug 即文件夹路径（`learning/flash-attention`），顺序取文件名前缀数字。
- `content/article-index.json` 的 `series` 条目是可选的标题/描述覆盖：

```json
"series": {
  "learning/flash-attention": {
    "title": "Flash Attention",
    "description": "从算法原理到 GPU 实现。"
  }
}
```

不写 `series` 条目时，系列标题回退为文件夹名。文章记录中**不要**再填写
`series`/`seriesOrder` 字段（构建会报错提示）。

构建会自动生成 `/series` 系列总览，以及每个系列文件夹自身的详情页
（如 `/learning/flash-attention/`）。文章底部有系列进度、上一篇和下一篇导航；
左侧目录会把系列文件夹标记为"系列"节点，并按 `seriesOrder` 排列文章。

- 推送到非 `main` 分支或打开 PR 时，`.github/workflows/ci.yml` 会运行类型检查、格式检查和 Quartz 构建。
- 推送到 `main` 时，`.github/workflows/deploy.yml` 会构建 `public/` 静态产物并发布到 GitHub Pages。
- 仓库的 Pages 发布源应设置为 `GitHub Actions`。
