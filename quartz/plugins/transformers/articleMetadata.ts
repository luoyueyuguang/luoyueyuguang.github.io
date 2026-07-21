import { slugTag } from "../../util/path"
import { QuartzTransformerPlugin } from "../types"

export interface ArticleMetadataEntry {
  title: string
  date: string
  tags: string[]
  series?: string
  seriesOrder?: number
}

export type ArticleMetadataIndex = Record<string, ArticleMetadataEntry>

export interface SeriesMetadataEntry {
  title: string
  description?: string
}

export type SeriesMetadataIndex = Record<string, SeriesMetadataEntry>

export interface ContentMetadataIndex {
  series: SeriesMetadataIndex
  articles: ArticleMetadataIndex
}

interface Options {
  index: ContentMetadataIndex
}

function validateArticleIndex(index: ContentMetadataIndex) {
  const { articles, series } = index
  for (const [slug, metadata] of Object.entries(series)) {
    if (!slug || slug.startsWith("/") || slugTag(slug) !== slug) {
      throw new Error(`Invalid series slug "${slug}" in article-index.json.`)
    }

    if (!metadata.title?.trim()) {
      throw new Error(`Series "${slug}" is missing a title in article-index.json.`)
    }
  }

  const seriesOrders = new Map<string, Map<number, string>>()
  for (const [slug, metadata] of Object.entries(articles)) {
    if (!slug || slug.startsWith("/") || slug.endsWith(".md") || slug.endsWith("/index")) {
      throw new Error(
        `Invalid article slug "${slug}" in article-index.json. Use a content-relative slug without .md.`,
      )
    }

    if (!metadata.title?.trim()) {
      throw new Error(`Article "${slug}" is missing a title in article-index.json.`)
    }

    if (!metadata.date || Number.isNaN(new Date(metadata.date).getTime())) {
      throw new Error(`Article "${slug}" has an invalid date in article-index.json.`)
    }

    if (!Array.isArray(metadata.tags) || metadata.tags.length === 0) {
      throw new Error(`Article "${slug}" must have at least one tag in article-index.json.`)
    }

    const hasSeries = metadata.series !== undefined
    const hasSeriesOrder = metadata.seriesOrder !== undefined
    if (hasSeries !== hasSeriesOrder) {
      throw new Error(
        `Article "${slug}" must define both series and seriesOrder in article-index.json.`,
      )
    }

    if (metadata.series) {
      if (!series[metadata.series]) {
        throw new Error(`Article "${slug}" references unknown series "${metadata.series}".`)
      }

      if (!Number.isInteger(metadata.seriesOrder) || metadata.seriesOrder! < 1) {
        throw new Error(`Article "${slug}" must use a positive integer seriesOrder.`)
      }

      const orders = seriesOrders.get(metadata.series) ?? new Map<number, string>()
      const duplicate = orders.get(metadata.seriesOrder!)
      if (duplicate) {
        throw new Error(
          `Articles "${duplicate}" and "${slug}" both use order ${metadata.seriesOrder} in series "${metadata.series}".`,
        )
      }
      orders.set(metadata.seriesOrder!, slug)
      seriesOrders.set(metadata.series, orders)
    }
  }
}

export const ArticleMetadata: QuartzTransformerPlugin<Options> = (opts) => {
  const index = opts?.index ?? { series: {}, articles: {} }
  const { articles, series } = index
  validateArticleIndex(index)

  return {
    name: "ArticleMetadata",
    markdownPlugins({ allSlugs }) {
      const knownSlugs: Set<string> = new Set(allSlugs)
      const missingArticles = Object.keys(articles).filter((slug) => !knownSlugs.has(slug))
      if (missingArticles.length > 0) {
        throw new Error(
          `article-index.json references missing Markdown files: ${missingArticles.join(", ")}`,
        )
      }

      return [
        () => {
          return (_tree, file) => {
            const metadata = articles[file.data.slug!]
            if (!metadata) return

            const tags = [
              ...new Set(metadata.tags.map((tag) => slugTag(tag.trim())).filter(Boolean)),
            ]
            const seriesMetadata = metadata.series ? series[metadata.series] : undefined
            file.data.frontmatter = {
              ...file.data.frontmatter,
              title: metadata.title.trim(),
              date: metadata.date,
              tags,
              article: true,
              series: metadata.series,
              seriesTitle: seriesMetadata?.title.trim(),
              seriesDescription: seriesMetadata?.description?.trim(),
              seriesOrder: metadata.seriesOrder,
            }
          }
        },
      ]
    },
  }
}
