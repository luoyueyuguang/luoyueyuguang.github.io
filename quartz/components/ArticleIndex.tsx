import { FullSlug, resolveRelative } from "../util/path"
import { Date, getDate } from "./Date"
import { byDateAndAlphabetical } from "./PageList"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/articleIndex.scss"
// @ts-ignore
import script from "./scripts/articleIndex.inline"

export default (() => {
  const ArticleIndex: QuartzComponent = (props: QuartzComponentProps) => {
    const { cfg, fileData, allFiles } = props
    if (!fileData.frontmatter?.articleIndex) return null

    const articles = allFiles
      .filter((page) => page.frontmatter?.article)
      .sort(byDateAndAlphabetical(cfg))

    const tagCounts = new Map<string, number>()
    for (const article of articles) {
      for (const tag of article.frontmatter?.tags ?? []) {
        tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
      }
    }
    const tags = [...tagCounts.keys()].sort((a, b) => a.localeCompare(b))

    return (
      <section class="article-index" data-article-index>
        <div class="article-index-controls">
          <label for="article-index-search">搜索文章</label>
          <input
            id="article-index-search"
            type="search"
            autocomplete="off"
            placeholder="输入标题或标签…"
            data-article-search
          />
          <div class="article-tag-filters" role="group" aria-label="按标签筛选">
            <button type="button" class="active" data-tag-filter="" aria-pressed="true">
              全部 <span>{articles.length}</span>
            </button>
            {tags.map((tag) => (
              <button type="button" data-tag-filter={tag} aria-pressed="false">
                #{tag} <span>{tagCounts.get(tag)}</span>
              </button>
            ))}
          </div>
          <div class="article-index-summary">
            <p data-result-count aria-live="polite">
              共 {articles.length} 篇文章
            </p>
            <a class="internal" href={resolveRelative(fileData.slug!, "tags/index" as FullSlug)}>
              查看标签归档
            </a>
          </div>
        </div>

        <ul class="article-index-list">
          {articles.map((article) => {
            const title = article.frontmatter!.title
            const articleTags = article.frontmatter?.tags ?? []
            const series = article.frontmatter?.series
            return (
              <li
                class="article-index-item"
                data-article-item
                data-title={title.toLocaleLowerCase()}
                data-tags={JSON.stringify(articleTags)}
                data-series={(article.frontmatter?.seriesTitle ?? series ?? "").toLocaleLowerCase()}
              >
                <div class="article-index-main">
                  <h2>
                    <a class="internal" href={resolveRelative(fileData.slug!, article.slug!)}>
                      {title}
                    </a>
                  </h2>
                  {article.dates && (
                    <p class="meta">
                      <Date date={getDate(cfg, article)!} locale={cfg.locale} />
                    </p>
                  )}
                  {series && (
                    <p class="article-series-reference">
                      <a
                        class="internal"
                        href={resolveRelative(fileData.slug!, `series/${series}` as FullSlug)}
                      >
                        系列：{article.frontmatter?.seriesTitle} · 第{" "}
                        {article.frontmatter?.seriesOrder} 篇
                      </a>
                    </p>
                  )}
                </div>
                <ul class="tags">
                  {articleTags.map((tag) => (
                    <li>
                      <a
                        class="internal tag-link"
                        href={resolveRelative(fileData.slug!, `articles?tag=${tag}` as FullSlug)}
                      >
                        {tag}
                      </a>
                    </li>
                  ))}
                </ul>
              </li>
            )
          })}
        </ul>
        <p class="article-index-empty" data-empty-state hidden>
          没有符合当前条件的文章。
        </p>
      </section>
    )
  }

  ArticleIndex.css = style
  ArticleIndex.afterDOMLoaded = script
  return ArticleIndex
}) satisfies QuartzComponentConstructor
