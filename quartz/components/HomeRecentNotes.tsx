import { FullSlug, resolveRelative } from "../util/path"
import { QuartzPluginData } from "../plugins/vfile"
import { Date, getDate } from "./Date"
import { byDateAndAlphabetical } from "./PageList"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/listPage.scss"

interface Options {
  title: string
  limit?: number
  include: (file: QuartzPluginData) => boolean
}

const defaultOptions: Options = {
  title: "最新文章",
  include: (file) => {
    return file.frontmatter?.article === true
  },
}

export default ((userOpts?: Partial<Options>) => {
  const HomeRecentNotes: QuartzComponent = (props: QuartzComponentProps) => {
    const { cfg, fileData, allFiles } = props
    if (!fileData.frontmatter?.autoRecent) {
      return null
    }

    const opts = { ...defaultOptions, ...userOpts }
    const pages = allFiles
      .filter(opts.include)
      .filter((page) => page.dates)
      .sort(byDateAndAlphabetical(cfg))
    const limitedPages = opts.limit ? pages.slice(0, opts.limit) : pages

    return (
      <section class="home-recent-notes">
        <div class="home-recent-header">
          <h2>{opts.title}</h2>
          <nav aria-label="文章浏览入口">
            <a class="internal" href={resolveRelative(fileData.slug!, "series/index" as FullSlug)}>
              文章系列
            </a>
            <a class="internal" href={resolveRelative(fileData.slug!, "articles" as FullSlug)}>
              全部文章与标签 →
            </a>
          </nav>
        </div>
        <ul class="section-ul">
          {limitedPages.map((page) => {
            const title = page.frontmatter?.title
            const tags = page.frontmatter?.tags ?? []

            return (
              <li class="section-li">
                <div class="section">
                  <div>
                    {page.dates && (
                      <p class="meta">
                        <Date date={getDate(cfg, page)!} locale={cfg.locale} />
                      </p>
                    )}
                  </div>
                  <div class="desc">
                    <h3>
                      <a href={resolveRelative(fileData.slug!, page.slug!)} class="internal">
                        {title}
                      </a>
                    </h3>
                  </div>
                  <ul class="tags">
                    {tags.map((tag) => (
                      <li>
                        <a
                          class="internal tag-link"
                          href={resolveRelative(fileData.slug!, `tags/${tag}` as FullSlug)}
                        >
                          {tag}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              </li>
            )
          })}
        </ul>
      </section>
    )
  }

  HomeRecentNotes.css =
    style +
    `
.home-recent-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
}

.home-recent-header > h2 {
  margin-bottom: 0;
}

.home-recent-header > nav {
  display: flex;
  gap: 1rem;
}

@media (max-width: 800px) {
  .home-recent-header {
    align-items: flex-start;
    flex-direction: column;
    gap: 0.4rem;
  }

  .home-recent-header > nav {
    flex-wrap: wrap;
    gap: 0.4rem 1rem;
  }
}
`
  return HomeRecentNotes
}) satisfies QuartzComponentConstructor
