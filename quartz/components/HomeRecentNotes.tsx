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
    const slug = file.slug ?? ""
    return slug.includes("/") && !slug.endsWith("/index")
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
        <h2>{opts.title}</h2>
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

  HomeRecentNotes.css = style
  return HomeRecentNotes
}) satisfies QuartzComponentConstructor
