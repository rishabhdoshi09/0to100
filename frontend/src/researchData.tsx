import { useEffect, useMemo, useState } from 'react'
import { fetchJson } from './http'

const reportBase = `${window.location.protocol}//${window.location.hostname}:8766`

type LinkItem = { label: string; url: string; official: string }
type Requirement = {
  key: string
  label: string
  status: string
  available: boolean
  source: string
  as_of: string
  age_days: number | null
  max_age_days: number
  why: string
  instructions: string
  accepted_extensions: string[]
  template_available: boolean
  template_url: string
  links: LinkItem[]
  latest_upload: Record<string, unknown>
  acquisition?: string
  source_url?: string
  source_date?: string
  acquired_at?: string
  parser?: string
  sha256?: string
  evidence?: string
  sources_attempted?: Array<{ url?: string; ok?: boolean; error?: string; path?: string }>
  failure_reason?: string
}
type RuntimeSource = {
  key: string
  label: string
  status: string
  available: boolean
  as_of: string
  age_days: number | null
  max_age_days: number
}
type UploadItem = {
  evidence_id: string
  kind: string
  filename: string
  as_of: string
  uploaded_at: string
  source_url: string
  structured: boolean
  extracted: boolean
  sha256: string
  bytes: number
}
type EvidenceStatus = {
  symbol: string
  generated_at: string
  coverage_pct: number
  requirements: Requirement[]
  runtime_sources: RuntimeSource[]
  uploads: UploadItem[]
  raw_fundamentals: {
    available: boolean
    fetched_at: string
    age_days: number | null
    freshness: string
    sections: Record<string, boolean | number>
  }
}
type Draft = { file: File | null; asOf: string; sourceUrl: string }

const today = () => new Date().toISOString().slice(0, 10)
const statusClass = (status: string, acquisition?: string) => {
  if (acquisition === 'AUTO_SOURCED' && status === 'FRESH') return 'evidence-status fresh'
  if (acquisition === 'AUTO_SOURCED') return 'evidence-status unknown'
  if (acquisition === 'AUTOMATION_FAILED' || status === 'AUTOMATION_FAILED') return 'evidence-status missing'
  if (status === 'FRESH') return 'evidence-status fresh'
  if (status === 'STALE') return 'evidence-status stale'
  if (status === 'UNKNOWN_DATE') return 'evidence-status unknown'
  return 'evidence-status missing'
}

const acquisitionLabel = (item: Requirement, sourcing = false) => {
  if (sourcing && item.acquisition !== 'AUTO_SOURCED' && item.acquisition !== 'MANUAL') {
    return 'AUTO-SOURCING'
  }
  if (item.acquisition === 'AUTO_SOURCED') return 'ACQUIRED'
  if (item.acquisition === 'AUTOMATION_FAILED') return 'FAILED — MANUAL FALLBACK AVAILABLE'
  if (item.acquisition === 'MANUAL') return 'FAILED — MANUAL FALLBACK AVAILABLE'
  if (item.acquisition === 'MISSING') return 'AUTO-SOURCING'
  return item.status
}
const humanBytes = (value: number) => {
  if (!value) return '0 B'
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

export function ResearchDataView({ symbol }: { symbol: string }) {
  const [status, setStatus] = useState<EvidenceStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(false)
  const [acquiring, setAcquiring] = useState(false)
  const [acquireNote, setAcquireNote] = useState('')
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})

  const load = async () => {
    if (!symbol) {
      setStatus(null)
      return
    }
    setLoading(true)
    try {
      const payload = await fetchJson<EvidenceStatus>(`${reportBase}/evidence/${encodeURIComponent(symbol)}`, {
        headers: { Accept: 'application/json' },
        timeoutMs: 20_000,
      })
      setStatus(payload)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Evidence service unavailable')
    } finally {
      setLoading(false)
    }
  }

  const autoAcquire = async () => {
    if (!symbol) return
    setAcquiring(true)
    setAcquireNote('QuantTerm is locating official sources…')
    setError('')
    try {
      const response = await fetch(
        `${reportBase}/evidence/${encodeURIComponent(symbol)}/actions/auto-acquire`,
        { method: 'POST', headers: { Accept: 'application/json' } },
      )
      if (!response.ok) throw new Error(await response.text())
      const payload = await response.json() as {
        auto_sourced?: number
        automation_failed?: number
        coverage?: EvidenceStatus
        steps?: Array<{ id?: string; ok?: boolean; error?: string; skipped?: boolean }>
      }
      if (payload.coverage) setStatus(payload.coverage)
      const failed = (payload.steps || []).filter((step) => step.ok === false && !step.skipped)
      setAcquireNote(
        `Automatic acquisition finished · ${payload.auto_sourced ?? 0} sourced · ${payload.automation_failed ?? 0} failed`
        + (failed.length ? ` · ${failed.map((step) => step.error || step.id).join('; ')}` : ''),
      )
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Automatic acquisition failed')
      setAcquireNote('AUTOMATION FAILED — manual upload remains available for the gaps.')
    } finally {
      setAcquiring(false)
    }
  }

  useEffect(() => {
    if (!symbol) return
    void load().then(() => { void autoAcquire() })
  }, [symbol])

  const missingCount = useMemo(() => status?.requirements.filter((item) => !item.available).length || 0, [status])
  const staleCount = useMemo(() => status?.requirements.filter((item) => item.status === 'STALE').length || 0, [status])
  const autoCount = useMemo(() => status?.requirements.filter((item) => item.acquisition === 'AUTO_SOURCED').length || 0, [status])
  const failedCount = useMemo(() => status?.requirements.filter((item) => item.acquisition === 'AUTOMATION_FAILED').length || 0, [status])
  const draft = (key: string): Draft => drafts[key] || { file: null, asOf: today(), sourceUrl: '' }
  const patchDraft = (key: string, patch: Partial<Draft>) => {
    setDrafts((current) => ({ ...current, [key]: { ...draft(key), ...patch } }))
  }

  const runAutomatic = async (action: 'fundamentals' | 'history' | 'news' | 'fno') => {
    setBusy(`auto-${action}`)
    setError('')
    try {
      const endpoint = action === 'fundamentals'
        ? `${reportBase}/evidence/${encodeURIComponent(symbol)}/actions/refresh-fundamentals`
        : `/api/controls/${action === 'history' ? 'REFRESH_DATA_NOW' : action === 'news' ? 'REFRESH_NEWS_NOW' : 'REFRESH_FNO_NOW'}`
      const response = await fetch(endpoint, { method: 'POST', headers: { Accept: 'application/json' } })
      if (!response.ok) throw new Error(await response.text())
      window.setTimeout(() => void load(), action === 'fundamentals' ? 100 : 1500)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `${action} refresh failed`)
    } finally {
      setBusy('')
    }
  }

  const upload = async (requirement: Requirement) => {
    const current = draft(requirement.key)
    if (!current.file) {
      setError(`Choose a file for ${requirement.label}`)
      return
    }
    if (!current.asOf) {
      setError(`Enter the source data date for ${requirement.label}`)
      return
    }
    setBusy(requirement.key)
    setError('')
    try {
      const query = new URLSearchParams({ as_of: current.asOf, source_url: current.sourceUrl })
      const response = await fetch(
        `${reportBase}/evidence/${encodeURIComponent(symbol)}/${encodeURIComponent(requirement.key)}?${query.toString()}`,
        {
          method: 'POST',
          headers: { 'Content-Type': current.file.type || 'application/octet-stream', 'X-Filename': current.file.name },
          body: current.file,
        },
      )
      if (!response.ok) throw new Error(await response.text())
      const payload = await response.json() as { status: EvidenceStatus }
      setStatus(payload.status)
      setDrafts((items) => ({ ...items, [requirement.key]: { file: null, asOf: today(), sourceUrl: '' } }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Upload failed')
    } finally {
      setBusy('')
    }
  }

  if (!symbol) {
    return <section className="research-data-view"><div className="evidence-empty"><h2>Select a stock first</h2><p>Choose a stock from Scanner, Long-Term, F&O Desk or search. QuantTerm will then show exactly which research datasets are available, stale or missing.</p></div></section>
  }

  return (
    <section className="research-data-view">
      {error && <div className="api-warning">{error} <button type="button" className="secondary" onClick={() => void load()}>Retry</button></div>}
      {loading && !status ? <p className="panel-copy">Loading evidence status…</p> : null}
      <div className="evidence-summary">
        <div><span>SYMBOL</span><strong>{symbol}</strong></div>
        <div><span>RESEARCH</span><strong>{autoCount}/{status?.requirements.length || 0} auto-sourced</strong></div>
        <div><span>AUTOMATION FAILED</span><strong>{failedCount}</strong></div>
        <div><span>MISSING / STALE</span><strong>{missingCount} / {staleCount}</strong></div>
        <div><span>DEEP FUNDAMENTALS</span><strong>{status?.raw_fundamentals.freshness || 'UNKNOWN'}</strong></div>
      </div>

      <div className="evidence-panel">
        <header>
          <div>
            <h2>Automatic data preparation</h2>
            <p>
              {acquiring
                ? 'AUTO-SOURCING official filings and datasets. Do not hunt annual reports unless a class fails.'
                : acquireNote || 'QuantTerm locates official sources itself. Hunt-and-upload is only shown after AUTO-SOURCING fails.'}
            </p>
          </div>
          <div className="resource-links">
            <button type="button" disabled={acquiring} onClick={() => void autoAcquire()}>{acquiring ? 'Acquiring…' : 'Acquire evidence now'}</button>
            <button type="button" onClick={() => void load()}>Refresh status</button>
          </div>
        </header>
        <div className="resource-links">
          <button type="button" disabled={busy === 'auto-fundamentals'} onClick={() => void runAutomatic('fundamentals')}>Fetch deep fundamentals</button>
          <button type="button" disabled={busy === 'auto-history'} onClick={() => void runAutomatic('history')}>Prepare official price history</button>
          <button type="button" disabled={busy === 'auto-news'} onClick={() => void runAutomatic('news')}>Refresh news and filings</button>
          <button type="button" disabled={busy === 'auto-fno'} onClick={() => void runAutomatic('fno')}>Refresh F&O instruments</button>
        </div>
      </div>

      <div className="evidence-panel">
        <header><div><h2>Runtime data dates</h2><p>These are the exact dates powering current QuantTerm views.</p></div></header>
        <div className="runtime-grid">
          {(status?.runtime_sources || []).map((item) => (
            <article key={item.key}>
              <span>{item.label}</span>
              <strong className={statusClass(item.status)}>{item.status}</strong>
              <small>As of {item.as_of || 'UNKNOWN'}</small>
              <small>{item.age_days === null ? 'Age unknown' : `${item.age_days} day(s) old`} · limit {item.max_age_days}d</small>
            </article>
          ))}
        </div>
      </div>

      <div className="evidence-panel">
        <header><div><h2>Research completion desk</h2><p>QuantTerm shows what it obtained. Manual upload appears only when automation missed or failed a class.</p></div></header>
        <div className="requirements-list">
          {(status?.requirements || []).map((item) => {
            const current = draft(item.key)
            const needsManual = item.acquisition === 'AUTOMATION_FAILED' || item.acquisition === 'MANUAL' || item.status === 'SOURCE_ATTACHED_UNPARSED'
            return (
              <article className="requirement-card" key={item.key}>
                <div className="requirement-head">
                  <div><h3>{item.label}</h3><p>{item.why}</p></div>
                  <div className={statusClass(item.status, item.acquisition)}>{acquisitionLabel(item, acquiring)}</div>
                </div>
                <div className="requirement-meta">
                  <span>Status <strong>{item.status}</strong></span>
                  <span>Source date <strong>{item.source_date || item.as_of || 'UNKNOWN'}</strong></span>
                  <span>Acquired at <strong>{item.acquired_at || '—'}</strong></span>
                  <span>Age <strong>{item.age_days === null ? 'unknown' : `${item.age_days}d`}</strong></span>
                  <span>Freshness limit <strong>{item.max_age_days}d</strong></span>
                  <span>Source <strong>{item.source || 'not loaded'}</strong></span>
                </div>
                {item.evidence ? <p className="requirement-instructions">{item.evidence}</p> : null}
                {item.source_url ? <p className="panel-copy">Source URL: <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a></p> : null}
                {item.sha256 ? <p className="panel-copy">Content hash {item.sha256.slice(0, 16)} · parser {item.parser || '—'}</p> : null}
                {item.acquisition === 'AUTO_SOURCED' ? (
                  <p className="requirement-instructions">ACQUIRED automatically. No operator hunt required.</p>
                ) : item.acquisition === 'AUTOMATION_FAILED' || item.acquisition === 'MANUAL' ? (
                  <p className="requirement-instructions">
                    FAILED — MANUAL FALLBACK AVAILABLE. Reason: {item.failure_reason || 'unknown'}.
                    {(item.sources_attempted || []).length
                      ? ` Sources attempted: ${(item.sources_attempted || []).map((row) => row.url || row.path || 'unknown').filter(Boolean).join(', ')}`
                      : ''}
                    {item.instructions ? ` ${item.instructions}` : ''}
                  </p>
                ) : acquiring ? (
                  <p className="requirement-instructions">AUTO-SOURCING this class. Manual upload stays hidden until it fails.</p>
                ) : (
                  <p className="requirement-instructions">Waiting for automatic acquisition. Do not hunt annual reports yet.</p>
                )}
                <div className="resource-links">
                  {item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">View source</a> : null}
                  {item.links.map((link) => <a key={link.url} href={link.url} target="_blank" rel="noreferrer">{link.official === 'true' ? 'Official · ' : ''}{link.label}</a>)}
                  {item.template_available && needsManual && <a href={`${reportBase}${item.template_url}`} target="_blank" rel="noreferrer">Download CSV template</a>}
                </div>
                {needsManual ? (
                  <>
                    <small className="accepted-files">Manual fallback · accepted: {item.accepted_extensions.join(', ')}</small>
                    <div className="upload-grid">
                      <label>Source data date<input type="date" value={current.asOf} onChange={(event) => patchDraft(item.key, { asOf: event.target.value })} /></label>
                      <label>Source URL<input type="url" placeholder="Paste official filing or IR link" value={current.sourceUrl} onChange={(event) => patchDraft(item.key, { sourceUrl: event.target.value })} /></label>
                      <label>Evidence file<input type="file" accept={item.accepted_extensions.join(',')} onChange={(event) => patchDraft(item.key, { file: event.target.files?.[0] || null })} /></label>
                      <button type="button" disabled={busy === item.key} onClick={() => void upload(item)}>{busy === item.key ? 'Uploading…' : 'Upload evidence'}</button>
                    </div>
                  </>
                ) : null}
              </article>
            )
          })}
        </div>
      </div>

      <div className="evidence-panel">
        <header><div><h2>Uploaded source ledger</h2><p>Every file carries its original name, as-of date, checksum and extraction status.</p></div></header>
        {status?.uploads.length ? (
          <div className="upload-ledger">
            {status.uploads.map((item) => (
              <article key={`${item.evidence_id}-${item.kind}`}>
                <div><strong>{item.kind.replaceAll('_', ' ')}</strong><span>{item.filename}</span></div>
                <div><span>As of {item.as_of}</span><span>{humanBytes(item.bytes)}</span></div>
                <div><span>{item.extracted ? 'Structured and readable' : 'Source attached; extraction pending'}</span><code>{item.sha256.slice(0, 12)}</code></div>
                <a href={`${reportBase}/evidence/${encodeURIComponent(symbol)}/files/${item.evidence_id}`} target="_blank" rel="noreferrer">Open uploaded file</a>
              </article>
            ))}
          </div>
        ) : <div className="evidence-empty"><p>No user evidence uploaded for {symbol}.</p></div>}
      </div>
    </section>
  )
}
