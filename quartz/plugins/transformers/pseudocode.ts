import { QuartzTransformerPlugin } from "../types"
import type { Element, ElementContent, Root } from "hast"
import { h } from "hastscript"
import { visit } from "unist-util-visit"

/**
 * ```pseudocode 渲染成论文里算法列表的样子。
 *
 * 论文的算法用 algorithm2e / algorithmic 排版：外面的框、左上角的
 * "Algorithm N: 标题"、左侧行号栏、关键字加粗、注释右置。这里复刻这套版式，
 * 而不是把伪代码当普通代码块用等宽字体平铺——伪代码是散文的一种排版，
 * 不是待编译的源码。
 *
 * 作者写法（行号自动生成，不要在正文里手写）：
 *
 *   ```pseudocode title="Algorithm 1: FlashAttention 前向"
 *   for j in 1..T_c:
 *       S = Q K^T          # 注释
 *   ```
 *
 * 缩进按块内最小缩进量归一化，所以 2 空格和 4 空格都能用。
 */

const KEYWORDS = [
  "for",
  "foreach",
  "while",
  "repeat",
  "until",
  "do",
  "end",
  "if",
  "then",
  "else",
  "elif",
  "elseif",
  "return",
  "function",
  "procedure",
  "break",
  "continue",
  "in",
  "to",
  "downto",
  "and",
  "or",
  "not",
  "each",
]

const KEYWORD_RE = new RegExp(`\\b(${KEYWORDS.join("|")})\\b`, "g")
// 注释起始符：▷ 与 // / # 都是通用写法。
// `←` 不算注释符——它在算法记法里是赋值号（`O_i ← diag(ℓ)^{-1} O_i`），
// 把它当注释会把赋值语句截断。要写注释用 # 或 //。
const COMMENT_RE = /(▷|\/\/|#)/
// 作者若沿用"3. "这类手写行号，去掉它——行号由渲染器生成
const LEADING_NUMBER_RE = /^\s*\d+\s*[.:]\s+/

/**
 * 读 hast 属性。hast 会把 `data-language` 这类名字规范化成 `dataLanguage`
 * （序列化时才转回连字符写法），所以两种拼法都要认，否则属性永远读不到。
 */
function prop(el: Element, name: string): unknown {
  const properties = el.properties ?? {}
  const camel = name.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase())
  return properties[camel] ?? properties[name]
}

const isElement = (node: unknown, tag: string): node is Element =>
  (node as Element | undefined)?.type === "element" && (node as Element).tagName === tag

interface AlgoLine {
  level: number
  code: string
  comment: string
}

function toLines(raw: string): AlgoLine[] {
  const src = raw
    .replace(/\t/g, "    ")
    .split("\n")
    // 空行不参与渲染：行号是逐行递增的，留一条空行会得到一个和相邻行号重叠的空行号。
    // 算法列表本来也不必用空行分组——函数/过程名本身就是分界。
    .filter((line) => line.trim() !== "")

  // 缩进单位取块内最小的非零缩进量，2 空格与 4 空格两种习惯都能容纳
  let unit = Number.POSITIVE_INFINITY
  for (const line of src) {
    if (line.trim() === "") continue
    const indent = line.length - line.trimStart().length
    if (indent > 0) unit = Math.min(unit, indent)
  }
  if (!Number.isFinite(unit)) unit = 2

  return src.map((line) => {
    const indent = line.length - line.trimStart().length
    let body = line.trimEnd()
    let comment = ""

    // 注释符前面必须是空白，避免把 `C#` 这类标识符当注释
    const m = COMMENT_RE.exec(body)
    if (m?.index !== undefined) {
      const before = m.index === 0 ? "" : body[m.index - 1]
      if (m.index === 0 || /\s/.test(before)) {
        comment = body.slice(m.index + m[0].length).trim()
        body = body.slice(0, m.index).trimEnd()
      }
    }

    return {
      level: Math.round(indent / unit),
      code: body.replace(LEADING_NUMBER_RE, "").trim(),
      comment,
    }
  })
}

/** 把关键字切成 <span class="algo-kw">，其余原样。 */
function codeNodes(code: string): ElementContent[] {
  const out: ElementContent[] = []
  let last = 0
  for (const m of code.matchAll(new RegExp(KEYWORD_RE.source, "g"))) {
    const at = m.index ?? 0
    if (at > last) out.push({ type: "text", value: code.slice(last, at) })
    out.push(h("span.algo-kw", m[0]))
    last = at + m[0].length
  }
  if (last < code.length) out.push({ type: "text", value: code.slice(last) })
  return out
}

function plainText(node: Element): string {
  let text = ""
  visit(node, "text", (t) => {
    text += t.value
  })
  return text.trim()
}

/** 找到该节点里的伪代码 <code>；不是伪代码则返回 null。 */
function pseudocodeOf(node: Element): Element | null {
  const pre = node.tagName === "pre" ? node : node.children.find((c) => isElement(c, "pre"))
  const code = pre && isElement(pre, "pre") ? pre.children.find((c) => isElement(c, "code")) : null
  return code && prop(code, "data-language") === "pseudocode" ? code : null
}

export const Pseudocode: QuartzTransformerPlugin = () => {
  return {
    name: "Pseudocode",
    htmlPlugins() {
      return [
        () => (tree: Root) => {
          // 在 figure 这一层替换：rehype-pretty-code 把代码块包进
          // <figure><figcaption>…</figcaption><pre>…</pre></figure>，
          // 要连同标题一起换掉，就必须拿到 figure 自身在父节点里的位置。
          visit(tree, "element", (node: Element, index, parent) => {
            const inFigure = node.tagName === "figure"
            const barePre = node.tagName === "pre" && !isElement(parent, "figure")
            if (!inFigure && !barePre) return

            const code = pseudocodeOf(node)
            if (!code) return

            // shiki 对未知语言降级为纯文本，每行一个 <span data-line>
            const lines = toLines(plainText(code))

            let caption = ""
            if (inFigure) {
              const cap = node.children.find((c) => isElement(c, "figcaption"))
              // 单独再判一次：find 的返回值不会因回调里的守卫而收窄
              if (
                cap &&
                isElement(cap, "figcaption") &&
                prop(cap, "data-rehype-pretty-code-title") !== undefined
              ) {
                caption = plainText(cap)
              }
            }

            const items = lines.map((line) => {
              const children: ElementContent[] = []
              if (line.code) children.push(h("span.algo-code", ...codeNodes(line.code)))
              if (line.comment) children.push(h("span.algo-comment", line.comment))
              return h("li.algo-line", { style: `--lvl:${line.level}` }, ...children)
            })

            const rendered = h(
              "figure.algo",
              {},
              caption ? h("figcaption.algo-caption", caption) : null,
              h("ol.algo-lines", ...items),
            )

            if (index !== undefined && parent) parent.children.splice(index, 1, rendered)
          })
        },
      ]
    },
  }
}
