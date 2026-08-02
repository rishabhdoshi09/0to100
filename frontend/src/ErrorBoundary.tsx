import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error: Error | null }

/**
 * Never leave retailers on a silent white/black screen.
 * Module or render failures become a visible recovery panel.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep console evidence for local debugging; do not invent product state.
    console.error('QuantTerm UI crashed', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="ui-crash-panel" role="alert">
        <h1>QuantTerm UI hit an error</h1>
        <p>
          The page crashed while rendering. This is a frontend bug, not missing market data.
          Copy the detail below, then hard-refresh or restart the stack.
        </p>
        <pre>{this.state.error.message}</pre>
        <div className="inline-actions">
          <button type="button" onClick={() => window.location.reload()}>Hard refresh</button>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
          >
            Try render again
          </button>
        </div>
        <small>
          Restart tip: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh
        </small>
      </div>
    )
  }
}
