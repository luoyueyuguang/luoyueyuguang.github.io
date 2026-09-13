interface PyodideInstance {
  loadedPackages?: Record<string, unknown>
  loadPackage(name: string): Promise<void>
  runPythonAsync(code: string): Promise<unknown>
}
declare var loadPyodide: ((opts: { indexURL: string }) => Promise<PyodideInstance>) | undefined

type RunPre = HTMLElement & { __runBound?: boolean }

const PYODIDE_VERSION = "0.26.4"
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

let pyodidePromise: Promise<PyodideInstance> | null = null
// serialize interpreter access so concurrent runs don't clobber each other's state
let runChain: Promise<void> = Promise.resolve()

function loadPyodideScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script")
    s.src = `${PYODIDE_BASE}pyodide.js`
    s.onload = () => resolve()
    s.onerror = () => reject(new Error("Pyodide 加载失败，请检查网络"))
    document.head.appendChild(s)
  })
}

async function getPyodide(): Promise<PyodideInstance> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      if (typeof loadPyodide === "undefined") await loadPyodideScript()
      return loadPyodide!({ indexURL: PYODIDE_BASE })
    })()
  }
  return pyodidePromise
}

async function runPython(code: string): Promise<string> {
  const pyodide = await getPyodide()
  // Preload the packages the snippet imports; Pyodide ships both.
  const needed: string[] = []
  if (/\bnumpy\b|\bnp\./.test(code)) needed.push("numpy")
  if (/\bmpmath\b/.test(code)) needed.push("mpmath")
  for (const pkg of needed) {
    if (!pyodide.loadedPackages?.[pkg]) await pyodide.loadPackage(pkg)
  }
  // Redirect stdout/stderr via Python-level StringIO so print() keeps its newlines.
  const wrapped = [
    "import sys, io",
    "__o = io.StringIO(); __e = io.StringIO()",
    "sys.stdout = __o; sys.stderr = __e",
    code,
    "__o.getvalue() + __e.getvalue()",
  ].join("\n")

  try {
    const out = await pyodide.runPythonAsync(wrapped)
    return String(out ?? "").trimEnd()
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    // code may have printed before raising; grab partial stdout
    const partial = await pyodide
      .runPythonAsync("__o.getvalue()")
      .then((v) => String(v ?? "").trimEnd())
      .catch(() => "")
    return (partial ? partial + "\n" : "") + msg
  }
}

// 取块的源码。优先按 [data-line] 逐行拼接：它不依赖布局，块即使处在折叠容器里
// 也能取到；innerText 在未渲染的元素上会返回空串。两种方式在站点现有的 87 个
// python 块上逐块比对过，结果完全一致。
function blockSource(codeBlock: HTMLElement): string {
  const lines = codeBlock.querySelectorAll(":scope > span[data-line]")
  if (lines.length > 0) {
    return Array.from(lines)
      .map((l) => l.textContent ?? "")
      .join("\n")
  }
  return codeBlock.innerText.replace(/\n\n/g, "\n")
}

// Build a run button + output panel for one Python code block.
function setupRunBlock(codeBlock: HTMLElement): void {
  const pre = codeBlock.closest("pre") as RunPre | null
  if (!pre || pre.__runBound) return
  pre.__runBound = true

  const button = document.createElement("button")
  button.className = "run-button"
  button.type = "button"
  button.textContent = "▶ 运行"
  button.ariaLabel = "运行这段 Python 代码"

  const editButton = document.createElement("button")
  editButton.className = "run-button"
  editButton.type = "button"
  editButton.textContent = "✎ 编辑"
  editButton.ariaLabel = "修改这段代码后再运行"

  const resetButton = document.createElement("button")
  resetButton.className = "run-button"
  resetButton.type = "button"
  resetButton.textContent = "↺ 还原"
  resetButton.ariaLabel = "恢复成原文的代码"
  resetButton.hidden = true

  const toolbar = document.createElement("div")
  toolbar.className = "run-toolbar"
  toolbar.append(button, editButton, resetButton)

  const output = document.createElement("div")
  output.className = "code-output"
  output.hidden = true

  const header = document.createElement("div")
  header.className = "code-output-header"
  const title = document.createElement("span")
  title.className = "code-output-title"
  title.textContent = "输出"
  const status = document.createElement("span")
  status.className = "code-output-status"
  status.textContent = "就绪"
  const clear = document.createElement("button")
  clear.className = "code-output-clear"
  clear.type = "button"
  clear.textContent = "清空"
  header.append(title, status, clear)

  const body = document.createElement("pre")
  body.className = "code-output-body"
  output.append(header, body)
  pre.after(output)

  const original = blockSource(codeBlock)
  // 高亮后的 <code> 带内联 display:grid，所以隐藏它要连内联样式一起换掉，
  // 不能只靠 [hidden]（内联声明优先于 UA 样式表）。改完原样还回去。
  const codeStyle = codeBlock.getAttribute("style")
  let editor: HTMLTextAreaElement | null = null
  let editing = false

  const getEditor = (): HTMLTextAreaElement => {
    if (editor) return editor
    editor = document.createElement("textarea")
    editor.className = "code-editor"
    editor.spellcheck = false
    editor.autocapitalize = "off"
    editor.setAttribute("autocorrect", "off")
    editor.ariaLabel = "编辑这段 Python 代码"
    editor.value = original
    editor.rows = original.split("\n").length
    editor.addEventListener("input", () => {
      editor!.rows = editor!.value.split("\n").length
      // 内容与原文一致时"还原"没有意义，就收起来
      resetButton.hidden = editor!.value === original
    })
    editor.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Tab") {
        e.preventDefault()
        const el = editor!
        el.setRangeText("    ", el.selectionStart, el.selectionEnd, "end")
      } else if (e.key === "Escape") {
        setEditing(false)
      }
    })
    codeBlock.after(editor)
    return editor
  }

  const setEditing = (on: boolean) => {
    editing = on
    const ed = on ? getEditor() : editor
    if (on) {
      codeBlock.setAttribute("style", "display:none")
    } else if (codeStyle === null) {
      codeBlock.removeAttribute("style")
    } else {
      codeBlock.setAttribute("style", codeStyle)
    }
    if (ed) ed.hidden = !on
    pre.classList.toggle("editing", on)
    editButton.textContent = on ? "✓ 完成" : "✎ 编辑"
    if (on) ed?.focus()
  }

  // 编辑过之后一直用编辑器里的内容：收起编辑器再点运行，跑的仍是改过的代码。
  const onClick = async () => {
    if (button.disabled) return
    button.disabled = true
    const label = button.textContent
    button.textContent = "运行中…"
    output.hidden = false
    output.classList.remove("code-output-error")
    status.textContent = "运行中"
    status.classList.add("running")
    body.textContent = ""
    const source = editor ? editor.value : original
    runChain = runChain.then(async () => {
      try {
        const result = await runPython(source)
        body.textContent = result || "（无输出）"
        status.textContent = "完成"
      } catch (e: unknown) {
        body.textContent = e instanceof Error ? e.message : String(e)
        output.classList.add("code-output-error")
        status.textContent = "出错"
      } finally {
        status.classList.remove("running")
        button.disabled = false
        button.textContent = label
      }
    })
  }

  const onClear = () => {
    body.textContent = ""
    output.hidden = true
    output.classList.remove("code-output-error")
    status.textContent = "就绪"
    status.classList.remove("running")
  }

  const onEdit = () => setEditing(!editing)
  const onReset = () => {
    if (!editor) return
    editor.value = original
    editor.rows = original.split("\n").length
    resetButton.hidden = true
  }

  button.addEventListener("click", onClick)
  clear.addEventListener("click", onClear)
  editButton.addEventListener("click", onEdit)
  resetButton.addEventListener("click", onReset)
  window.addCleanup(() => {
    button.removeEventListener("click", onClick)
    clear.removeEventListener("click", onClear)
    editButton.removeEventListener("click", onEdit)
    resetButton.removeEventListener("click", onReset)
  })
  pre.prepend(toolbar)
}

document.addEventListener("nav", () => {
  // data-runnable="false" 由构建期的 RunnableMarking 标注：块要么导入运行器没有的包，
  // 要么是用 `...` 省略的源码摘录。这类块给按钮只会让人点出一串报错。
  const blocks = document.querySelectorAll(
    'pre > code[data-language="python"]:not([data-runnable="false"])',
  )
  for (const el of Array.from(blocks)) {
    if (el instanceof HTMLElement) setupRunBlock(el)
  }
})
