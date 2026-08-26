// @ts-ignore
import clipboardScript from "./scripts/clipboard.inline"
import clipboardStyle from "./styles/clipboard.scss"
// @ts-ignore
import runCodeScript from "./scripts/runCode.inline"
import runCodeStyle from "./styles/runCode.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Body: QuartzComponent = ({ children }: QuartzComponentProps) => {
  return <div id="quartz-body">{children}</div>
}

Body.afterDOMLoaded = clipboardScript + "\n" + runCodeScript
Body.css = clipboardStyle + runCodeStyle

export default (() => Body) satisfies QuartzComponentConstructor
