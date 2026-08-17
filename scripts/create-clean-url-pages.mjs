import { mkdir, readFile, readdir, writeFile } from "node:fs/promises"
import path from "node:path"

const outputDir = "public"

// Quartz 生成相对路径资源引用（如 ../index.css、../assets/x.svg、../learning/MSA）。
// clean-URL 副本位于比原文件深一层的目录，所有相对引用需加深一级。
const RELATIVE_URL_RE = /(href|src)="([^"]*)"/g

function isRelativeUrl(url) {
  if (!url) return false
  if (url.startsWith("#")) return false
  // 绝对路径（/index.css）与带协议/协议的引用（http:, mailto:, data:, //cdn…）保持原样
  if (url.startsWith("/") || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(url)) return false
  return true
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)))
      continue
    }

    files.push(fullPath)
  }

  return files
}

const htmlFiles = (await walk(outputDir)).filter((file) => {
  const relative = path.relative(outputDir, file)
  const basename = path.basename(file)
  return file.endsWith(".html") && basename !== "index.html" && relative !== "404.html"
})

for (const file of htmlFiles) {
  const cleanDir = file.slice(0, -".html".length)
  const cleanIndex = path.join(cleanDir, "index.html")

  const html = await readFile(file, "utf8")
  const rewritten = html.replace(RELATIVE_URL_RE, (match, attr, url) =>
    isRelativeUrl(url) ? `${attr}="../${url}"` : match,
  )

  await mkdir(cleanDir, { recursive: true })
  await writeFile(cleanIndex, rewritten)
}

console.log(`Created ${htmlFiles.length} clean URL page copies`)
