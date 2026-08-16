import { useEffect, useMemo, useState } from 'react'
import {
  fetchDataCoverage,
  fetchDataJobs,
  fetchDataProviders,
  runDataJob,
  fetchStockFundamentals,
  type DataCoveragePayload,
  type DataJobsPayload,
  type DataProvidersPayload,
} from './productApi'

const reportBase = import.meta.env.DEV
  ? ''
  : `${window.location.protocol}//${window.location.hostname}:8766`

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
  example_available?: boolean
  example_url?: string
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
type Draft = { file: File | null; asOf: string; sourceUrl: string }

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

type ResolveStep = {
  step: number
  source: string
  status: string
  message: string
  elapsed_ms?: number
  coverage?: number
  reputed?: boolean
  official?: boolean
}
type NextAction = { label: string; url: string; kind: string }

export function ResearchDataView({ symbol }: { symbol: string }) {
  const [status, setStatus] = useState<EvidenceStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [fundaBusy, setFundaBusy] = useState(false)
  const [resolveTrail, setResolveTrail] = useState<ResolveStep[]>([])
  const [nextActions, setNextActions] = useState<NextAction[]>([])
  const [resolveSource, setResolveSource] = useState('')
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})
  const [providers, setProviders] = useState<DataProvidersPayload | null>(null)
  const [jobs, setJobs] = useState<DataJobsPayload | null>(null)
  const [symbolCoverage, setSymbolCoverage] = useState<DataCoveragePayload | null>(null)
  const [universeCoverage, setUniverseCoverage] = useState<DataCoveragePayload | null>(null)
  const [jobBusy, setJobBusy] = useState('')

  const runPlatformJob = async (jobId: string) => {
    setJobBusy(jobId)
    setError('')
    try {
      const result = await runDataJob(jobId)
      if (!result.ok) throw new Error(result.error || result.message || 'Job failed')
      await loadPlatformData()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Job run failed')
    } finally {
      setJobBusy('')
    }
  }

  const loadPlatformData = async () => {
    const [prov, jobList, symCov, uniCov] = await Promise.all([
      fetchDataProviders().catch(() => null),
      fetchDataJobs().catch(() => null),
      symbol ? fetchDataCoverage(symbol).catch(() => null) : Promise.resolve(null),
      fetchDataCoverage().catch(() => null),
    ])
    setProviders(prov)
    setJobs(jobList)
    setSymbolCoverage(symCov)
    setUniverseCoverage(uniCov)
  }

  const loadEvidence = async () => {
    if (!symbol) {
      setStatus(null)
      return
    }
    try {
      const response = await fetch(`${reportBase}/evidence/${encodeURIComponent(symbol)}`, { headers: { Accept: 'application/json' } })
      if (!response.ok) throw new Error(await response.text())
      setStatus(await response.json() as EvidenceStatus)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Evidence service unavailable')
    }
  }

  const loadFundamentals = async (force: boolean) => {
    if (!symbol) return
    setFundaBusy(true)
    if (!force) setError('')
    try {
      const payload = await fetchStockFundamentals(symbol, force)
      setResolveTrail(payload.steps || [])
      setNextActions(payload.next_actions || [])
      setResolveSource(payload.source || '')
      await loadEvidence()
      if (!payload.accepted) {
        setError(payload.message || 'Fundamentals missing after all sources — use next actions below')
      } else if (force) {
        setError('')
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Fundamentals fetch failed — use Retry')
    } finally {
      setFundaBusy(false)
    }
  }

  useEffect(() => {
    if (!symbol) {
      setStatus(null)
      setSymbolCoverage(null)
      setResolveTrail([])
      setNextActions([])
      setResolveSource('')
      return
    }
    void loadFundamentals(false)
    void loadPlatformData()
  }, [symbol])

  const missingCount = useMemo(() => status?.requirements.filter((item) => !item.available).length || 0, [status])
  const staleCount = useMemo(() => status?.requirements.filter((item) => item.status === 'STALE').length || 0, [status])
  const draft = (key: string): Draft => drafts[key] || { file: null, asOf: today(), sourceUrl: '' }
  const patchDraft = (key: string, patch: Partial<Draft>) => {
    setDrafts((current) => ({ ...current, [key]: { ...draft(key), ...patch } }))
  }

  const runAutomatic = async (action: 'fundamentals' | 'history' | 'news' | 'fno') => {
    setBusy(`auto-${action}`)
    setError('')
    try {
      if (action === 'fundamentals') {
        await loadFundamentals(true)
      } else {
        const endpoint = `/api/controls/${action === 'history' ? 'REFRESH_DATA_NOW' : action === 'news' ? 'REFRESH_NEWS_NOW' : 'REFRESH_FNO_NOW'}`
        const response = await fetch(endpoint, { method: 'POST', headers: { Accept: 'application/json' } })
        if (!response.ok) throw new Error(await response.text())
        window.setTimeout(() => void loadEvidence(), 1500)
      }
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

  const installWorkedExample = async () => {
    setBusy('worked-example')
    setError('')
    try {
      const response = await fetch(
        `${reportBase}/evidence/${encodeURIComponent(symbol)}/actions/install-worked-example`,
        { method: 'POST', headers: { Accept: 'application/json' } },
      )
      if (!response.ok) throw new Error(await response.text())
      const payload = await response.json() as { status: EvidenceStatus; note?: string }
      setStatus(payload.status)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Worked-example install failed')
    } finally {
      setBusy('')
    }
  }

  if (!symbol) {
    return <section className="research-data-view"><div className="evidence-empty"><h2>Select a stock first</h2><p>Open a name from Ideas or search. This tab is the file layer under that stock — which datasets are fresh, stale or missing.</p></div></section>
  }

  return (
    <section className="research-data-view">
      {error && (
        <div className="api-warning">
          {error}
          <button type="button" className="mode-action" disabled={fundaBusy} onClick={() => void loadFundamentals(true)}>
            {fundaBusy ? 'Retrying…' : 'Retry fundamentals'}
          </button>
        </div>
      )}
      {fundaBusy && !error && (
        <div className="api-warning" style={{ borderColor: 'var(--accent-cyan, #26d7ff)' }}>
          Resolving fundamentals for {symbol} — Screener.in → Yahoo Finance → cache → uploads…
        </div>
      )}
      {resolveTrail.length > 0 && (
        <div className="evidence-panel">
          <header>
            <div>
              <h2>Resolve trail</h2>
              <p>Every source attempt yields a status. {resolveSource ? `Active source: ${resolveSource}.` : 'No source produced usable data yet.'}</p>
            </div>
            <button type="button" disabled={fundaBusy} onClick={() => void loadFundamentals(true)}>
              {fundaBusy ? 'Resolving…' : 'Re-resolve now'}
            </button>
          </header>
          <div className="runtime-grid">
            {resolveTrail.map((step) => (
              <article key={`${step.step}-${step.source}-${step.status}`}>
                <span>#{step.step} · {step.source}</span>
                <strong className={statusClass(step.status === 'OK' ? 'FRESH' : step.status === 'ERROR' || step.status === 'EXHAUSTED' ? 'MISSING' : 'STALE')}>
                  {step.status}
                </strong>
                <small>{step.message}</small>
                <small>{typeof step.elapsed_ms === 'number' ? `${step.elapsed_ms} ms` : ''}{typeof step.coverage === 'number' ? ` · coverage ${step.coverage}` : ''}</small>
              </article>
            ))}
          </div>
          {nextActions.length > 0 && (
            <div className="resource-links" style={{ marginTop: '0.75rem' }}>
              {nextActions.map((action) => (
                <a
                  key={`${action.kind}-${action.url}`}
                  href={action.url.startsWith('/') ? `${reportBase}${action.url}` : action.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {action.kind === 'official' ? 'Official · ' : action.kind === 'reputed' ? 'Reputed · ' : ''}{action.label}
                </a>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="evidence-summary">
        <div><span>SYMBOL</span><strong>{symbol}</strong></div>
        <div><span>RESEARCH COVERAGE</span><strong>{status?.coverage_pct ?? 0}%</strong></div>
        <div><span>MISSING DATASETS</span><strong>{missingCount}</strong></div>
        <div><span>STALE DATASETS</span><strong>{staleCount}</strong></div>
        <div><span>DEEP FUNDAMENTALS</span><strong>{status?.raw_fundamentals.freshness || 'UNKNOWN'}</strong></div>
      </div>

      <div className="evidence-panel">
        <header>
          <div><h2>Data platform audit</h2><p>Provider registry, refresh jobs, and per-symbol coverage from /api/data/* (not inferred from UI).</p></div>
          <button type="button" onClick={() => void loadPlatformData()}>Refresh platform</button>
        </header>
        {providers && (
          <div className="runtime-grid">
            {providers.providers.map((row) => (
              <article key={row.name}>
                <span>{row.name}</span>
                <strong className={statusClass(row.status)}>{row.status}</strong>
                <small>{row.coverage_note}</small>
                <small>Auth: {row.authentication_status} · caps: {row.capabilities.join(', ')}</small>
              </article>
            ))}
          </div>
        )}
        {jobs && (
          <div className="fno-table wide-table" style={{ marginTop: '12px' }}>
            <div className="fno-head"><span>JOB</span><span>CONTROL</span><span>DESCRIPTION</span><span>ACTION</span></div>
            {jobs.jobs.map((job) => (
              <div className="fno-row" key={job.id} style={{ display: 'grid', cursor: 'default' }}>
                <strong>{job.label}</strong>
                <span>{job.control || job.trigger}</span>
                <span>{job.description}</span>
                <button type="button" disabled={jobBusy === job.id} onClick={() => void runPlatformJob(job.id)}>
                  {jobBusy === job.id ? 'Running…' : 'Run'}
                </button>
              </div>
            ))}
          </div>
        )}
        {symbolCoverage?.coverage && (
          <div className="key-value-list" style={{ marginTop: '12px' }}>
            <div><span>{symbol} identity</span><strong>{String(symbolCoverage.coverage.identity ?? '—')}</strong></div>
            <div><span>Price history</span><strong>{String(symbolCoverage.coverage.price_history ?? '—')}</strong></div>
            <div><span>Fundamentals</span><strong>{String(symbolCoverage.coverage.fundamentals ?? '—')}</strong></div>
            <div><span>Ratios</span><strong>{String(symbolCoverage.coverage.ratios ?? '—')}</strong></div>
            <div><span>Long-term eligible</span><strong>{String(symbolCoverage.coverage.long_term_eligible ?? '—')}</strong></div>
          </div>
        )}
        {universeCoverage?.audited != null && (
          <p className="panel-copy" style={{ marginTop: '12px' }}>
            Universe sample: {universeCoverage.audited} symbols audited · remediation queue {universeCoverage.remediation_queue?.length ?? 0} items
            {universeCoverage.status_counts && (
              <> · counts: {Object.entries(universeCoverage.status_counts).map(([k, v]) => `${k}=${v}`).join(', ')}</>
            )}
          </p>
        )}
      </div>

      <div className="evidence-panel">
        <header>
          <div><h2>Automatic data preparation</h2><p>QuantTerm fetches what it can itself. These operations are independent of paper trading.</p></div>
          <button type="button" onClick={() => void loadEvidence()}>Refresh status</button>
        </header>
        <div className="resource-links">
          <button type="button" disabled={busy === 'auto-fundamentals' || fundaBusy} onClick={() => void runAutomatic('fundamentals')}>
            {fundaBusy || busy === 'auto-fundamentals' ? 'Fetching…' : 'Retry deep fundamentals'}
          </button>
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
        <header><div><h2>Research completion desk</h2><p>Open the source, download a template, or upload the original evidence with its data date.</p></div>
          <button type="button" disabled={busy === 'worked-example'} onClick={() => void installWorkedExample()}>
            {busy === 'worked-example' ? 'Installing worked example…' : 'Auto-install worked example'}
          </button>
        </header>
        <p className="requirement-instructions">
          When NSE/BSE/Screener pages are blocked or incomplete, use <strong>Download worked example</strong> (clickable CSV)
          or <strong>Auto-install worked example</strong> (download + upload + analysis in one step).
          These rows are schema-valid fixtures with example.com provenance — not live exchange fundamentals.
        </p>
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
                  {item.links.map((link) => <a key={link.url} href={link.url} target="_blank" rel="noreferrer">{link.official === 'true' ? 'Official · ' : ''}{link.label}</a>)}
                  {item.template_available && <a href={`${reportBase}${item.template_url}`} target="_blank" rel="noreferrer">Download CSV template</a>}
                  {item.example_available && item.example_url && (
                    <a href={`${reportBase}${item.example_url}`} target="_blank" rel="noreferrer">Download worked example</a>
                  )}
                </div>
                <small className="accepted-files">Accepted: {item.accepted_extensions.join(', ')}</small>
                <div className="upload-grid">
                  <label>Source data date<input type="date" value={current.asOf} onChange={(event) => patchDraft(item.key, { asOf: event.target.value })} /></label>
                  <label>Source URL<input type="url" placeholder="Paste official filing or IR link" value={current.sourceUrl} onChange={(event) => patchDraft(item.key, { sourceUrl: event.target.value })} /></label>
                  <label>Evidence file<input type="file" accept={item.accepted_extensions.join(',')} onChange={(event) => patchDraft(item.key, { file: event.target.files?.[0] || null })} /></label>
                  <button type="button" disabled={busy === item.key} onClick={() => void upload(item)}>{busy === item.key ? 'Uploading…' : 'Upload evidence'}</button>
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
