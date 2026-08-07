import { useEffect, useMemo, useState } from 'react'

type ReportPdfViewerProps = {
  open: boolean
  title: string
  viewUrl: string
  onClose: () => void
}

/** Build the optional download URL without changing the inline view URL. */
export function reportDownloadUrl(viewUrl: string): string {
  if (!viewUrl) return ''
  return viewUrl.includes('?') ? `${viewUrl}&download=true` : `${viewUrl}?download=true`
}

/**
 * In-terminal PDF viewer.
 *
 * Browsers (especially Safari) often force-download when an iframe navigates
 * directly to a PDF URL. Fetch as a blob and render via object URL so View is
 * always inline; Download remains an explicit optional action.
 */
export function ReportPdfViewer({ open, title, viewUrl, onClose }: ReportPdfViewerProps) {
  const [objectUrl, setObjectUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const downloadHref = useMemo(() => reportDownloadUrl(viewUrl), [viewUrl])
  const safeFilename = useMemo(() => {
    const base = title.replace(/[^\w.\-]+/g, '_').replace(/^_+|_+$/g, '') || 'quantterm-report'
    return base.toLowerCase().endsWith('.pdf') ? base : `${base}.pdf`
  }, [title])

  useEffect(() => {
    if (!open || !viewUrl) {
      setObjectUrl('')
      setLoading(false)
      setError('')
      return
    }

    let cancelled = false
    let createdUrl = ''
    setLoading(true)
    setError('')
    setObjectUrl('')

    void (async () => {
      try {
        const response = await fetch(viewUrl, { headers: { Accept: 'application/pdf' } })
        if (!response.ok) {
          const detail = await response.text().catch(() => '')
          throw new Error(detail || `Report failed (${response.status})`)
        }
        const buffer = await response.arrayBuffer()
        const blob = new Blob([buffer], { type: 'application/pdf' })
        createdUrl = URL.createObjectURL(blob)
        if (cancelled) {
          URL.revokeObjectURL(createdUrl)
          return
        }
        setObjectUrl(createdUrl)
      } catch (reason) {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : 'Could not open report in the terminal')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => {
      cancelled = true
      if (createdUrl) URL.revokeObjectURL(createdUrl)
    }
  }, [open, viewUrl])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

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
            <p>Opens inside QuantTerm. Download only if you want a local copy.</p>
          </div>
          <div className="report-pdf-actions">
            <a
              className="mode-action"
              href={objectUrl || downloadHref}
              download={safeFilename}
              aria-disabled={!objectUrl && !downloadHref}
            >
              Download PDF
            </a>
            <button type="button" onClick={onClose}>Close</button>
          </div>
        </header>
        {loading && (
          <div className="report-pdf-status" role="status">Generating report for on-screen viewing…</div>
        )}
        {error && !loading && (
          <div className="report-pdf-status report-pdf-error" role="alert">
            <strong>Could not display this report in the terminal.</strong>
            <p>{error}</p>
            <p>You can still use Download PDF, or retry after the report API is healthy on :8766.</p>
          </div>
        )}
        {objectUrl && !loading && !error && (
          <iframe className="report-pdf-frame" title={title} src={objectUrl} />
        )}
      </section>
    </div>
  )
}
