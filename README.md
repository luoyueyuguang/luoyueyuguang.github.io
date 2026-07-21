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

系列用于组织具有固定阅读顺序的多篇文章。先在 `series` 中定义一次系列：

```json
"series": {
  "flash-attention": {
    "title": "Flash Attention",
    "description": "从算法原理到 GPU 实现。"
  }
}
```

然后在属于该系列的文章记录中填写系列 slug 和篇序：

```json
"learning/flash-attention-2": {
  "title": "Flash Attention 2：分块与重计算",
  "date": "2026-07-21",
  "tags": ["learning", "attention", "cuda"],
  "series": "flash-attention",
  "seriesOrder": 2
}
```

构建会自动生成 `/series` 系列总览、`/series/flash-attention` 系列详情，以及文章底部的
系列进度、上一篇和下一篇导航。左侧目录还会把同一栏目中的系列文章收进可展开的系列
节点，并按照 `seriesOrder` 排列。同一系列内的 `seriesOrder` 必须是唯一的正整数。

- 推送到非 `main` 分支或打开 PR 时，`.github/workflows/ci.yml` 会运行类型检查、格式检查和 Quartz 构建。
- 推送到 `main` 时，`.github/workflows/deploy.yml` 会构建 `public/` 静态产物并发布到 GitHub Pages。
- 仓库的 Pages 发布源应设置为 `GitHub Actions`。
