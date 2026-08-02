import { depthLabel, type DisplayDepth } from './productLanguage'

export function DisplayDepthToggle({
  depth,
  onChange,
}: {
  depth: DisplayDepth
  onChange: (value: DisplayDepth) => void
}) {
  return (
    <div className="display-depth-toggle" role="group" aria-label="Display depth">
      <button
        type="button"
        className={depth === 'simple' ? 'active' : ''}
        onClick={() => onChange('simple')}
      >
        {depthLabel('simple')}
      </button>
      <button
        type="button"
        className={depth === 'professional' ? 'active' : ''}
        onClick={() => onChange('professional')}
      >
        {depthLabel('professional')}
      </button>
    </div>
  )
}
