document.addEventListener("nav", () => {
  const root = document.querySelector("[data-article-index]") as HTMLElement | null
  if (!root) return

  const search = root.querySelector("[data-article-search]") as HTMLInputElement
  const items = [...root.querySelectorAll("[data-article-item]")] as HTMLElement[]
  const buttons = [...root.querySelectorAll("[data-tag-filter]")] as HTMLButtonElement[]
  const resultCount = root.querySelector("[data-result-count]") as HTMLElement
  const emptyState = root.querySelector("[data-empty-state]") as HTMLElement
  let activeTag = ""

  const applyFilters = () => {
    const query = search.value.trim().toLocaleLowerCase()
    let visibleCount = 0

    for (const item of items) {
      const title = item.dataset.title ?? ""
      const tags: string[] = JSON.parse(item.dataset.tags ?? "[]")
      const series = item.dataset.series ?? ""
      const matchesTag = activeTag === "" || tags.includes(activeTag)
      const matchesQuery =
        query === "" ||
        title.includes(query) ||
        series.includes(query) ||
        tags.some((tag) => tag.includes(query))
      const visible = matchesTag && matchesQuery
      item.hidden = !visible
      if (visible) visibleCount += 1
    }

    resultCount.textContent = `共 ${visibleCount} 篇文章`
    emptyState.hidden = visibleCount !== 0
  }

  const onSearch = () => applyFilters()
  search.addEventListener("input", onSearch)
  window.addCleanup(() => search.removeEventListener("input", onSearch))

  for (const button of buttons) {
    const onClick = () => {
      activeTag = button.dataset.tagFilter ?? ""
      for (const candidate of buttons) {
        const selected = candidate === button
        candidate.classList.toggle("active", selected)
        candidate.setAttribute("aria-pressed", selected.toString())
      }
      applyFilters()
    }

    button.addEventListener("click", onClick)
    window.addCleanup(() => button.removeEventListener("click", onClick))
  }
})
