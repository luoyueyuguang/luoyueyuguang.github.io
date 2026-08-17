import { FullSlug, resolveRelative } from "../../util/path"
import { Date, getDate } from "../Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/seriesPage.scss"
import { SeriesMetadataIndex } from "../../plugins/transformers/articleMetadata"

export default (() => {
  const SeriesContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { cfg, fileData, allFiles } = props
    const pageSlug = fileData.slug!
    const isOverview = pageSlug === "series/index"
    const seriesSlug = isOverview
      ? ""
      : pageSlug.endsWith("/index")
        ? pageSlug.slice(0, -"/index".length)
        : pageSlug

    // 系列信息来自注入的 frontmatter（文件夹自动推导），无需额外配置
    const seriesIndex: SeriesMetadataIndex = {}
    for (const file of allFiles) {
      const slug = file.frontmatter?.series
      if (!slug || seriesIndex[slug]) continue
      seriesIndex[slug] = {
        title: file.frontmatter?.seriesTitle ?? slug,
        description: file.frontmatter?.seriesDescription,
      }
    }

    if (!isOverview && !seriesIndex[seriesSlug]) {
      throw new Error(`Component "SeriesContent" tried to render a non-series page: ${pageSlug}`)
    }

    const articlesInSeries = (slug: string) =>
      allFiles
        .filter((file) => file.frontmatter?.article && file.frontmatter?.series === slug)
        .sort(
          (first, second) =>
            (first.frontmatter?.seriesOrder ?? 0) - (second.frontmatter?.seriesOrder ?? 0),
        )

    if (isOverview) {
      const entries = Object.entries(seriesIndex).sort(([, first], [, second]) =>
        first.title.localeCompare(second.title),
      )

      return (
        <div class="series-overview">
          <p class="series-count">共 {entries.length} 个系列</p>
          <div class="series-grid">
            {entries.map(([slug, metadata]) => {
              const articles = articlesInSeries(slug)
              return (
                <section class="series-card">
                  <h2>
                    <a class="internal" href={resolveRelative(pageSlug, slug as FullSlug)}>
                      {metadata.title}
                    </a>
                  </h2>
                  {metadata.description && <p>{metadata.description}</p>}
                  <p class="series-card-meta">{articles.length} 篇文章</p>
                </section>
              )
            })}
          </div>
        </div>
      )
    }

    const metadata = seriesIndex[seriesSlug]
    if (!metadata) {
      throw new Error(`Unknown series "${seriesSlug}".`)
    }

    const articles = articlesInSeries(seriesSlug)
    return (
      <div class="series-detail">
        {metadata.description && <p class="series-description">{metadata.description}</p>}
        <p class="series-count">本系列共 {articles.length} 篇文章，按建议阅读顺序排列。</p>
        <ol class="series-article-list">
          {articles.map((article) => (
            <li>
              <span class="series-order">第 {article.frontmatter?.seriesOrder} 篇</span>
              <div class="series-article-main">
                <h2>
                  <a class="internal" href={resolveRelative(pageSlug, article.slug!)}>
                    {article.frontmatter?.title}
                  </a>
                </h2>
                {article.dates && (
                  <p class="meta">
                    <Date date={getDate(cfg, article)!} locale={cfg.locale} />
                  </p>
                )}
              </div>
              <ul class="tags">
                {(article.frontmatter?.tags ?? []).map((tag) => (
                  <li>
                    <a
                      class="internal tag-link"
                      href={resolveRelative(pageSlug, `articles?tag=${tag}` as FullSlug)}
                    >
                      {tag}
                    </a>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </div>
    )
  }

  SeriesContent.css = style
  return SeriesContent
}) satisfies QuartzComponentConstructor
