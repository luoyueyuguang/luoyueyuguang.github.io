document.addEventListener("nav", () => {
  const root = document.querySelector("[data-article-index]") as HTMLElement | null
  if (!root) return

  const search = root.querySelector("[data-article-search]") as HTMLInputElement
  const items = [...root.querySelectorAll("[data-article-item]")] as HTMLElement[]
  const buttons = [...root.querySelectorAll("[data-tag-filter]")] as HTMLButtonElement[]
  const resultCount = root.querySelector("[data-result-count]") as HTMLElement
  const emptyState = root.querySelector("[data-empty-state]") as HTMLElement
  let activeTag = new URLSearchParams(window.location.search).get("tag") ?? ""

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

  const setActiveTag = (tag: string, button: HTMLButtonElement | null) => {
    activeTag = tag
    for (const candidate of buttons) {
      const selected = candidate === button
      candidate.classList.toggle("active", selected)
      candidate.setAttribute("aria-pressed", selected.toString())
    }
    applyFilters()
  }

  // 从 URL ?tag= 参数初始化选中状态（点标签跳转进来）
  if (activeTag !== "") {
    const match = buttons.find((button) => button.dataset.tagFilter === activeTag)
    setActiveTag(activeTag, match ?? null)
  }

  const onSearch = () => applyFilters()
  search.addEventListener("input", onSearch)
  window.addCleanup(() => search.removeEventListener("input", onSearch))

  for (const button of buttons) {
    const onClick = () => {
      const tag = button.dataset.tagFilter ?? ""
      setActiveTag(tag, button)
      // 同步 URL，便于分享/刷新保持筛选状态
      const url = new URL(window.location.href)
      if (tag) {
        url.searchParams.set("tag", tag)
      } else {
        url.searchParams.delete("tag")
      }
      history.replaceState(null, "", url)
    }

    button.addEventListener("click", onClick)
    window.addCleanup(() => button.removeEventListener("click", onClick))
  }
})
