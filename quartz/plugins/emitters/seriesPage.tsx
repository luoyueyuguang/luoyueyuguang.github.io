import { QuartzComponentProps } from "../../components/types"
import { SeriesContent } from "../../components"
import { FullPageLayout } from "../../cfg"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import BodyConstructor from "../../components/Body"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { FilePath, FullSlug, joinSegments, pathToRoot } from "../../util/path"
import DepGraph from "../../depgraph"
import { SeriesMetadataIndex } from "../transformers/articleMetadata"
import { QuartzEmitterPlugin } from "../types"
import { defaultProcessedContent } from "../vfile"
import { write } from "./helpers"

interface SeriesPageOptions extends FullPageLayout {}

export const SeriesPage: QuartzEmitterPlugin<Partial<SeriesPageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultListPageLayout,
    pageBody: SeriesContent(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, left, right, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  return {
    name: "SeriesPage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async getDependencyGraph(ctx, content, _resources) {
      const graph = new DepGraph<FilePath>()
      for (const [_tree, file] of content) {
        const seriesSlug = file.data.frontmatter?.series
        if (!seriesSlug) continue

        graph.addEdge(
          file.data.filePath!,
          joinSegments(ctx.argv.output, "series", "index.html") as FilePath,
        )
        graph.addEdge(
          file.data.filePath!,
          joinSegments(ctx.argv.output, seriesSlug, "index.html") as FilePath,
        )
      }
      return graph
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const allFiles = content.map((item) => item[1].data)

      // 系列由文件夹自动推导：收集所有文章 frontmatter 中的 series
      const seriesFromFiles: SeriesMetadataIndex = {}
      for (const file of allFiles) {
        const slug = file.frontmatter?.series
        if (!slug || seriesFromFiles[slug]) continue
        seriesFromFiles[slug] = {
          title: file.frontmatter?.seriesTitle ?? slug,
          description: file.frontmatter?.seriesDescription,
        }
      }

      const pages = ["index", ...Object.keys(seriesFromFiles)]
      const emitted: FilePath[] = []

      for (const page of pages) {
        // 系列详情页位于其文件夹 URL（如 learning/flash-attention/），总览页在 /series/
        const slug =
          page === "index"
            ? (joinSegments("series", "index") as FullSlug)
            : (joinSegments(page, "index") as FullSlug)
        const metadata = page === "index" ? undefined : seriesFromFiles[page]
        const [tree, file] = defaultProcessedContent({
          slug,
          frontmatter: {
            title: metadata?.title ?? "文章系列",
            description: metadata?.description,
            tags: [],
          },
          description: metadata?.description,
        })
        const externalResources = pageResources(pathToRoot(slug), resources)
        const componentData: QuartzComponentProps = {
          ctx,
          fileData: file.data,
          externalResources,
          cfg: ctx.cfg.configuration,
          children: [],
          tree,
          allFiles,
        }

        emitted.push(
          await write({
            ctx,
            content: renderPage(
              ctx.cfg.configuration,
              slug,
              componentData,
              opts,
              externalResources,
            ),
            slug,
            ext: ".html",
          }),
        )
      }

      return emitted
    },
  }
}
