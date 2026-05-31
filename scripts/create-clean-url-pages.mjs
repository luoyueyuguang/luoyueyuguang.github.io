import { copyFile, mkdir, readdir } from "node:fs/promises"
import path from "node:path"

const outputDir = "public"
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

  await mkdir(cleanDir, { recursive: true })
  await copyFile(file, cleanIndex)
}

console.log(`Created ${htmlFiles.length} clean URL page copies`)
