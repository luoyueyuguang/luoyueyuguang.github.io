import { FullSlug, resolveRelative } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/seriesNavigation.scss"

export default (() => {
  const SeriesNavigation: QuartzComponent = ({ fileData, allFiles }: QuartzComponentProps) => {
    const series = fileData.frontmatter?.series
    if (!series || !fileData.frontmatter?.article) return null

    const articles = allFiles
      .filter((file) => file.frontmatter?.article && file.frontmatter?.series === series)
      .sort(
        (first, second) =>
          (first.frontmatter?.seriesOrder ?? 0) - (second.frontmatter?.seriesOrder ?? 0),
      )
    const currentIndex = articles.findIndex((article) => article.slug === fileData.slug)
    if (currentIndex === -1) return null

    const previous = articles[currentIndex - 1]
    const next = articles[currentIndex + 1]
    const title = fileData.frontmatter.seriesTitle ?? series

    return (
      <section class="series-navigation" aria-label={`${title} 系列导航`}>
        <div class="series-navigation-header">
          <div>
            <p class="series-navigation-label">所属系列</p>
            <h2>
              <a class="internal" href={resolveRelative(fileData.slug!, series as FullSlug)}>
                {title}
              </a>
            </h2>
          </div>
          <p>
            第 {fileData.frontmatter.seriesOrder} 篇 · 共 {articles.length} 篇
          </p>
        </div>
        {(previous || next) && (
          <nav class="series-navigation-links">
            <div>
              {previous && (
                <a class="internal" href={resolveRelative(fileData.slug!, previous.slug!)}>
                  ← 上一篇：{previous.frontmatter?.title}
                </a>
              )}
            </div>
            <div>
              {next && (
                <a class="internal" href={resolveRelative(fileData.slug!, next.slug!)}>
                  下一篇：{next.frontmatter?.title} →
                </a>
              )}
            </div>
          </nav>
        )}
      </section>
    )
  }

  SeriesNavigation.css = style
  return SeriesNavigation
}) satisfies QuartzComponentConstructor
