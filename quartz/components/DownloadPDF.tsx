// @ts-ignore
import script from "./scripts/downloadPdf.inline"
import style from "./styles/downloadPdf.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const DownloadPDF: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  // 首页只是个目录入口，不提供 PDF 下载
  if (fileData.slug === "index") return null

  return (
    <div class="download-pdf-wrap">
      <button class="download-pdf" type="button" aria-label="下载 PDF 版本">
        <svg
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M12 3a1 1 0 0 1 1 1v9.59l2.3-2.3a1 1 0 1 1 1.4 1.42l-4 4a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.42l2.3 2.3V4a1 1 0 0 1 1-1z" />
          <path d="M5 15a1 1 0 0 1 1 1v2a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-2a1 1 0 1 1 2 0v2a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4v-2a1 1 0 0 1 1-1z" />
        </svg>
        下载 PDF
      </button>
    </div>
  )
}

DownloadPDF.css = style
DownloadPDF.afterDOMLoaded = script

export default (() => DownloadPDF) satisfies QuartzComponentConstructor
