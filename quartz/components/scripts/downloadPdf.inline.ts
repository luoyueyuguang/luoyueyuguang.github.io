function bindDownloadPdf(): void {
  document.querySelectorAll<HTMLButtonElement>(".download-pdf").forEach((btn) => {
    // nav 事件在初始加载时也会触发一次，用标记避免重复绑定
    if (btn.dataset.pdfBound) return
    btn.dataset.pdfBound = "true"
    btn.addEventListener("click", () => {
      // 原生打印对话框中选择“另存为 PDF”即可导出
      window.print()
    })
  })
}

bindDownloadPdf()
document.addEventListener("nav", bindDownloadPdf)
