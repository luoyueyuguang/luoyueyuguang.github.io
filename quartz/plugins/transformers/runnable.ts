import { QuartzTransformerPlugin } from "../types"
import type { Element, Root } from "hast"
import { visit } from "unist-util-visit"

/**
 * 给浏览器运行器跑不了的 python 块打上标记，运行器据此不挂"▶ 运行"按钮。
 *
 * 站点内置的运行器是 Pyodide 0.26.4，只带 NumPy 与 mpmath。有些 python 块本来就
 * 不是拿来跑的：
 *
 * - 导入 torch / triton / flash_attn 这类它没有的包；
 * - 是需要读的源码摘录，用 `...` 省略了参数或函数体——这种连语法都过不了，
 *   点了只会得到 SyntaxError，而读者要的其实是"读"而不是"跑"。
 *
 * 打完标记仍保留 python 语法高亮（摘录是真实源码，高亮有价值），只是不再
 * 承诺"可运行"。判断只在构建期做一次，不影响运行时。
 */

// Pyodide 0.26.4 + 站点 runner 里确实没有的包。用黑名单而非白名单：
// 白名单会误伤标准库，那才是真正的回归。
const UNAVAILABLE = [
  "torch",
  "transformer_engine",
  "triton",
  "flash_attn",
  "cupy",
  "numba",
  "tensorflow",
  "jax",
  "keras",
  "sklearn",
  "scipy",
  "pandas",
  "matplotlib",
  "PIL",
  "cv2",
  "onnxruntime",
]

const UNAVAILABLE_RE = new RegExp(
  `^\\s*(?:from|import)\\s+(${UNAVAILABLE.join("|")})(?:\\.|\\s|$)`,
  "m",
)

// `def f(a, b, ...)` / `class C(...)`：省略号占位符在参数表里不是合法语法
const PLACEHOLDER_SIG_RE = /^\s*(?:async\s+)?(?:def|class)\s+\w+\s*\([^)]*\B\.\.\.[^)]*\)/m

// Triton 内核：`tl.` 是 Triton 运行时 API，Pyodide 里没有。
// 这类块是拿来读的源码，给了按钮只会点出一串 NameError。
const TRITON_RE = /(^|\s)@triton\.|(^|[^\w.])tl\./m

function prop(el: Element, name: string): unknown {
  const properties = el.properties ?? {}
  const camel = name.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase())
  return properties[camel] ?? properties[name]
}

function textOf(node: Element): string {
  let text = ""
  visit(node, "text", (t) => {
    text += t.value
  })
  return text
}

export const RunnableMarking: QuartzTransformerPlugin = () => {
  return {
    name: "RunnableMarking",
    htmlPlugins() {
      return [
        () => (tree: Root, file) => {
          // frontmatter 里写 `runCode: false` 可整页关掉运行按钮。
          // 通篇都是源码摘录的页面（例如逐行读 Triton 实现那一篇）用它，
          // 比逐块判断可靠：摘录里的片段引用了未定义的变量，静态判断看不出来。
          const pageOff = file?.data?.frontmatter?.runCode === false

          visit(tree, "element", (node: Element) => {
            if (node.tagName !== "code") return
            if (prop(node, "data-language") !== "python") return

            const code = textOf(node)
            const unRunnable =
              pageOff ||
              UNAVAILABLE_RE.test(code) ||
              PLACEHOLDER_SIG_RE.test(code) ||
              TRITON_RE.test(code)

            if (unRunnable) {
              node.properties = { ...node.properties, dataRunnable: "false" }
            }
          })
        },
      ]
    },
  }
}
