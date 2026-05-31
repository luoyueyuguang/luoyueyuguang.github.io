# luoyue's Blog

这是一个使用 [Quartz](https://quartz.jzhao.xyz/) 生成的 GitHub Pages 博客。

## 本地预览

```bash
npm ci
npm run serve
```

## GitHub Pages

内容放在 `content/` 目录中。

- 推送到非 `main` 分支或打开 PR 时，`.github/workflows/ci.yml` 会运行类型检查、格式检查和 Quartz 构建。
- 推送到 `main` 时，`.github/workflows/deploy.yml` 会构建 `public/` 静态产物并发布到 GitHub Pages。
- 仓库的 Pages 发布源应设置为 `GitHub Actions`。
