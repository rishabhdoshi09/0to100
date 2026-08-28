import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'
import './truth.css'
import './views.css'
import './marketViews.css'
import './researchData.css'
import './productViews.css'
import './experience.css'
import './design-tokens.css'
import './radar.css'
import './reco.css'
import './recommendations.css'
import './reco-desk.css'

// Compatibility for two legacy Investigate button guards. Duplicate acquisition
// is already prevented authoritatively by the API, which returns the active job
// for the same symbol instead of queueing a second download. Keeping this binding
// explicit prevents a runtime ReferenceError while that UI guard is consolidated.
globalThis.acquiring = false

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
