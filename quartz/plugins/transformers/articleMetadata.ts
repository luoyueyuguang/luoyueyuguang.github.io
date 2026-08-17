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

// 系列 = 文件夹：文件夹内文章文件名带 "N-" 数字前缀即属于该系列。
// 系列 slug 为文件夹路径，顺序取前缀数字，JSON 中的 series 条目仅作为标题/描述覆盖。
const SERIES_PREFIX_RE = /^(\d+)-/

export function deriveSeries(
  slugs: Iterable<string>,
  jsonSeries: SeriesMetadataIndex,
): { series: SeriesMetadataIndex; membership: Map<string, { series: string; order: number }> } {
  const series: SeriesMetadataIndex = {}
  const membership = new Map<string, { series: string; order: number }>()

  for (const slug of slugs) {
    const parts = slug.split("/")
    if (parts.length < 2) continue
    const fileName = parts[parts.length - 1]
    const match = fileName.match(SERIES_PREFIX_RE)
    if (!match) continue

    const folder = parts.slice(0, -1).join("/")
    const order = parseInt(match[1], 10)
    membership.set(slug, { series: folder, order })

    if (!series[folder]) {
      const override = jsonSeries[folder]
      series[folder] = {
        title: override?.title ?? folder.split("/").pop() ?? folder,
        description: override?.description,
      }
    }
  }

  return { series, membership }
}

function validateArticleIndex(index: ContentMetadataIndex) {
  const { articles, series } = index
  for (const [slug, metadata] of Object.entries(series)) {
    if (!slug || slug.startsWith("/") || slug.endsWith("/") || slug.endsWith(".md")) {
      throw new Error(
        `Invalid series slug "${slug}" in article-index.json. Use a content-relative folder path.`,
      )
    }

    if (!metadata.title?.trim()) {
      throw new Error(`Series "${slug}" is missing a title in article-index.json.`)
    }
  }

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

    if (metadata.series !== undefined || metadata.seriesOrder !== undefined) {
      throw new Error(
        `Article "${slug}" in article-index.json: series and seriesOrder are derived from ` +
          `folder structure (folder + "N-" filename prefix); remove these fields.`,
      )
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

      const { series: derivedSeries, membership } = deriveSeries(allSlugs, series)

      // 校验：JSON 里声明的系列必须对应实际存在、带数字前缀文章的文件夹
      for (const slug of Object.keys(series)) {
        if (!derivedSeries[slug]) {
          throw new Error(
            `Series "${slug}" in article-index.json does not match a folder containing ` +
              `"N-" prefixed articles.`,
          )
        }
      }

      // 校验：系列文件夹内的每篇注册文章都必须有数字前缀
      for (const [slug] of Object.entries(articles)) {
        const folder = slug.split("/").slice(0, -1).join("/")
        if (derivedSeries[folder] && !membership.has(slug)) {
          throw new Error(
            `Article "${slug}" is inside series folder "${folder}" but lacks a "N-" filename prefix.`,
          )
        }
      }

      return [
        () => {
          return (_tree, file) => {
            const metadata = articles[file.data.slug!]
            if (!metadata) return

            const tags = [
              ...new Set(metadata.tags.map((tag) => slugTag(tag.trim())).filter(Boolean)),
            ]
            const member = membership.get(file.data.slug!)
            const seriesMetadata = member ? derivedSeries[member.series] : undefined
            file.data.frontmatter = {
              ...file.data.frontmatter,
              title: metadata.title.trim(),
              date: metadata.date,
              tags,
              article: true,
              series: member?.series,
              seriesTitle: seriesMetadata?.title.trim(),
              seriesDescription: seriesMetadata?.description?.trim(),
              seriesOrder: member?.order,
            }
          }
        },
      ]
    },
  }
}
