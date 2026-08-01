import { useEffect, useMemo, useState } from 'react'

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

type Draft = {
  file: File | null
  asOf: string
  sourceUrl: string
}

const today = () => new Date().toISOString().slice(0, 10)

const statusClass = (status: string) => {
  if (status === 'FRESH') return 'evidence-status fresh'
  if (status === 'STALE') return 'evidence-status stale'
  if (status === 'UNKNOWN_DATE') return 'evidence-status unknown'
  return 'evidence-status missing'
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
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})

  const load = async () => {
    if (!symbol) {
      setStatus(null)
      return
    }
    try {
      const response = await fetch(`${reportBase}/evidence/${encodeURIComponent(symbol)}`, {
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) throw new Error(await response.text())
      setStatus(await response.json() as EvidenceStatus)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Evidence service unavailable')
    }
  }

  useEffect(() => {
    void load()
  }, [symbol])

  const missingCount = useMemo(
    () => status?.requirements.filter((item) => !item.available).length || 0,
    [status],
  )
  const staleCount = useMemo(
    () => status?.requirements.filter((item) => item.status === 'STALE').length || 0,
    [status],
  )

  const draft = (key: string): Draft => drafts[key] || { file: null, asOf: today(), sourceUrl: '' }
  const patchDraft = (key: string, patch: Partial<Draft>) => {
    setDrafts((current) => ({ ...current, [key]: { ...draft(key), ...patch } }))
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
          headers: {
            'Content-Type': current.file.type || 'application/octet-stream',
            'X-Filename': current.file.name,
          },
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
    return (
      <section className="research-data-view">
        <div className="evidence-empty">
          <h2>Select a stock first</h2>
          <p>Choose a stock from Scanner, Long-Term, F&O Desk or search. QuantTerm will then show exactly which research datasets are available, stale or missing.</p>
        </div>
      </section>
    )
  }

  return (
    <section className="research-data-view">
      {error && <div className="api-warning">{error}</div>}
      <div className="evidence-summary">
        <div><span>SYMBOL</span><strong>{symbol}</strong></div>
        <div><span>RESEARCH COVERAGE</span><strong>{status?.coverage_pct ?? 0}%</strong></div>
        <div><span>MISSING DATASETS</span><strong>{missingCount}</strong></div>
        <div><span>STALE DATASETS</span><strong>{staleCount}</strong></div>
        <div><span>DEEP FUNDAMENTALS</span><strong>{status?.raw_fundamentals.freshness || 'UNKNOWN'}</strong></div>
      </div>

      <div className="evidence-panel">
        <header>
          <div><h2>Runtime data dates</h2><p>These are the exact dates powering current QuantTerm views.</p></div>
          <button type="button" onClick={() => void load()}>Refresh status</button>
        </header>
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
        <header><div><h2>Research completion desk</h2><p>Open the source, download a template, or upload the original evidence with its data date.</p></div></header>
        <div className="requirements-list">
          {(status?.requirements || []).map((item) => {
            const current = draft(item.key)
            return (
              <article className="requirement-card" key={item.key}>
                <div className="requirement-head">
                  <div><h3>{item.label}</h3><p>{item.why}</p></div>
                  <div className={statusClass(item.status)}>{item.status}</div>
                </div>
                <div className="requirement-meta">
                  <span>As of <strong>{item.as_of || 'UNKNOWN'}</strong></span>
                  <span>Age <strong>{item.age_days === null ? 'unknown' : `${item.age_days}d`}</strong></span>
                  <span>Freshness limit <strong>{item.max_age_days}d</strong></span>
                  <span>Source <strong>{item.source || 'not loaded'}</strong></span>
                </div>
                <p className="requirement-instructions">{item.instructions}</p>
                <div className="resource-links">
                  {item.links.map((link) => (
                    <a key={link.url} href={link.url} target="_blank" rel="noreferrer">
                      {link.official === 'true' ? 'Official · ' : ''}{link.label}
                    </a>
                  ))}
                  {item.template_available && (
                    <a href={`${reportBase}${item.template_url}`} target="_blank" rel="noreferrer">Download CSV template</a>
                  )}
                </div>
                <small className="accepted-files">Accepted: {item.accepted_extensions.join(', ')}</small>
                <div className="upload-grid">
                  <label>
                    Source data date
                    <input type="date" value={current.asOf} onChange={(event) => patchDraft(item.key, { asOf: event.target.value })} />
                  </label>
                  <label>
                    Source URL
                    <input type="url" placeholder="Paste official filing or IR link" value={current.sourceUrl} onChange={(event) => patchDraft(item.key, { sourceUrl: event.target.value })} />
                  </label>
                  <label>
                    Evidence file
                    <input type="file" accept={item.accepted_extensions.join(',')} onChange={(event) => patchDraft(item.key, { file: event.target.files?.[0] || null })} />
                  </label>
                  <button type="button" disabled={busy === item.key} onClick={() => void upload(item)}>
                    {busy === item.key ? 'Uploading…' : 'Upload evidence'}
                  </button>
                </div>
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
