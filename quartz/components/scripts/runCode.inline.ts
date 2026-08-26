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
  if (/\bnumpy\b|\bnp\b/.test(code)) {
    if (!pyodide.loadedPackages?.["numpy"]) await pyodide.loadPackage("numpy")
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

  const source = codeBlock.innerText.replace(/\n\n/g, "\n")
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

  button.addEventListener("click", onClick)
  clear.addEventListener("click", onClear)
  window.addCleanup(() => {
    button.removeEventListener("click", onClick)
    clear.removeEventListener("click", onClear)
  })
  pre.prepend(button)
}

document.addEventListener("nav", () => {
  const blocks = document.querySelectorAll('pre > code[data-language="python"]')
  for (const el of Array.from(blocks)) {
    if (el instanceof HTMLElement) setupRunBlock(el)
  }
})
