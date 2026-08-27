type ReportPdfViewerProps = {
  open: boolean
  title: string
  viewUrl: string
  onClose: () => void
}

export function ReportPdfViewer({ open, title, viewUrl, onClose }: ReportPdfViewerProps) {
  if (!open) return null

  const downloadUrl = viewUrl.includes('?')
    ? `${viewUrl}&download=true`
    : `${viewUrl}?download=true`

  return (
    <div className="report-pdf-backdrop" role="presentation" onClick={onClose}>
      <section
        className="report-pdf-viewer"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="report-pdf-header">
          <div>
            <h2>{title}</h2>
            <p>Rendered in the terminal — use Download only if you need a local copy.</p>
          </div>
          <div className="report-pdf-actions">
            <a className="mode-action" href={downloadUrl} download>Download PDF</a>
            <button type="button" onClick={onClose}>Close</button>
          </div>
        </header>
        <iframe className="report-pdf-frame" title={title} src={viewUrl} />
      </section>
    </div>
  )
}
