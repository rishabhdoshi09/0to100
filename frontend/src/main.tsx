import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './ErrorBoundary'
import './styles.css'
import './truth.css'
import './views.css'
import './marketViews.css'
import './researchData.css'
import './productViews.css'
import './experience.css'
import './design-tokens.css'
import './radar.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
