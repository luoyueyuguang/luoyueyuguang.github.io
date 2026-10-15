import style from "./styles/aiNotice.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

// 未经人工复核的 AI 生成笔记：在正文上方显示提示条。
// 是否显示由 frontmatter 的 aiGenerated 决定，人工撰写的笔记不加这个字段。
const AiNotice: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  if (fileData.frontmatter?.aiGenerated !== true) {
    return null
  }

  return (
    <div class="ai-notice" role="note">
      <p class="ai-notice-label">AI 生成，需人工复核（needs human review）</p>
      <p>本页由 AI 生成，尚未经过人工复核，内容可能有误。请以论文和官方文档为准。</p>
    </div>
  )
}

AiNotice.css = style

export default (() => AiNotice) satisfies QuartzComponentConstructor
