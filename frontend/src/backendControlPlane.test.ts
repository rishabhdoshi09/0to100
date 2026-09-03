import { describe, expect, it } from 'vitest'
import {
  FORBIDDEN_HOME_CONTROLS,
  SAFE_HOME_CONTROLS,
  checkSystemRows,
  filterSafeActions,
  hasSecretKey,
  isActivatingKey,
  isSafeHomeControl,
  laneAriaLabel,
  lanePrimaryAction,
  laneSecondaryActions,
  liveMoneyStillLocked,
  nothingNeeded,
  scrubTechnical,
  technicalLines,
  type SystemLane,
} from './backendControlPlane'

const readyData: SystemLane = {
  id: 'data',
  label: 'DATA',
  status: 'Ready',
  status_code: 'READY',
  summary: '2,300+ stocks have usable history.',
  what: 'Official NSE market data',
  meaning: 'You do not need to do anything.',
  current: 'Last updated: 2026-09-01',
  next: 'Market scan',
  secondary_actions: [{ label: 'Refresh', control: 'REFRESH_DATA_NOW' }],
  live_locked: true,
}

const workingData: SystemLane = {
  id: 'data',
  status: 'Working',
  current: 'Checking the latest market snapshot',
  next: 'Run the market scan',
  secondary_actions: [{ label: 'Refresh', control: 'REFRESH_DATA_NOW' }],
}

const waitingData: SystemLane = {
  id: 'data',
  status: 'Waiting',
  waiting_for: 'Official session 2026-09-01',
  meaning: 'This is a normal dependency.',
}

const needsYou: SystemLane = {
  id: 'paper_bot',
  status: 'Needs you',
  needs_user: true,
  primary_action: {
    label: 'Resume',
    control: 'RESUME_NEW_PAPER_ENTRIES',
  },
}

const optionalBroker: SystemLane = {
  id: 'zerodha',
  status: 'Optional login',
  status_code: 'CAPABILITY_OFFLINE',
  needs_user: false,
  optional_capability: true,
  blocks_autonomy: false,
  login_required: true,
  primary_action: {
    label: 'Login to Zerodha',
    kind: 'instruction',
    instruction: 'Login only when broker-live capability is wanted.',
  },
  secondary_actions: [{
    label: 'Login to Zerodha',
    kind: 'instruction',
    instruction: 'Login only when broker-live capability is wanted.',
  }],
}

const problemData: SystemLane = {
  id: 'data',
  status: 'Problem',
  last_failure_reason: 'DATA_PREPARE failed',
  primary_action: { label: 'Retry', control: 'REFRESH_DATA_NOW' },
}

const paperOn: SystemLane = {
  id: 'paper_bot',
  status: 'Ready',
  secondary_actions: [
    { label: 'Pause new entries', control: 'PAUSE_NEW_PAPER_ENTRIES' },
    { label: 'Run paper cycle now', control: 'RUN_CYCLE_NOW' },
  ],
}

const paperPaused: SystemLane = {
  id: 'paper_bot',
  status: 'Needs you',
  needs_user: true,
  primary_action: { label: 'Resume', control: 'RESUME_NEW_PAPER_ENTRIES' },
}

const learning: SystemLane = {
  id: 'learning',
  status: 'Working',
  secondary_actions: [{ label: 'Verify now', control: 'VERIFY_FORWARD_SOAK' }],
}

describe('READY lane contract', () => {
  it('is clickable and explains readiness without a primary action', () => {
    expect(laneAriaLabel('data', readyData)).toContain('View details')
    expect(laneAriaLabel('data', readyData)).toContain('Ready')
    expect(readyData.what).toBeTruthy()
    expect(readyData.current).toBeTruthy()
    expect(lanePrimaryAction(readyData)).toBeNull()
    expect(nothingNeeded(readyData)).toBe(true)
    expect(laneSecondaryActions(readyData).map((a) => a.control)).toEqual(['REFRESH_DATA_NOW'])
  })
})

describe('WORKING lane contract', () => {
  it('shows current + next and hides duplicate refresh', () => {
    expect(workingData.current).toBeTruthy()
    expect(workingData.next).toBeTruthy()
    expect(lanePrimaryAction(workingData)).toBeNull()
    expect(laneSecondaryActions(workingData)).toEqual([])
    expect(nothingNeeded(workingData)).toBe(true)
  })
})

describe('WAITING dependency contract', () => {
  it('explains the dependency', () => {
    expect(waitingData.waiting_for).toMatch(/Official session|Market data|Zerodha|later/i)
  })
})

describe('NEEDS YOU contract', () => {
  it('shows the one primary action', () => {
    const action = lanePrimaryAction(needsYou)
    expect(action?.label).toBe('Resume')
    expect(action?.control).toBe('RESUME_NEW_PAPER_ENTRIES')
    expect(nothingNeeded(needsYou)).toBe(false)
  })
})

describe('optional capability contract', () => {
  it('keeps broker login discoverable without turning it into operator attention', () => {
    expect(optionalBroker.needs_user).toBe(false)
    expect(optionalBroker.blocks_autonomy).toBe(false)
    expect(lanePrimaryAction(optionalBroker)).toBeNull()
    expect(nothingNeeded(optionalBroker)).toBe(true)
    expect(laneSecondaryActions(optionalBroker)[0]?.kind).toBe('instruction')
    expect(laneAriaLabel('zerodha', optionalBroker)).toContain('no action required')
  })
})

describe('PROBLEM contract', () => {
  it('maps Retry to the canonical data control', () => {
    expect(lanePrimaryAction(problemData)?.control).toBe('REFRESH_DATA_NOW')
    expect(SAFE_HOME_CONTROLS.has('REFRESH_DATA_NOW')).toBe(true)
  })
})

describe('PAPER BOT contract', () => {
  it('maps pause and resume to existing controls', () => {
    expect(laneSecondaryActions(paperOn).map((a) => a.control)).toContain('PAUSE_NEW_PAPER_ENTRIES')
    expect(lanePrimaryAction(paperPaused)?.control).toBe('RESUME_NEW_PAPER_ENTRIES')
  })
})

describe('LEARNING contract', () => {
  it('verify uses the canonical verifier and is not a primary on a working lane', () => {
    expect(lanePrimaryAction(learning)).toBeNull()
    expect(laneSecondaryActions(learning).map((a) => a.control)).toEqual(['VERIFY_FORWARD_SOAK'])
  })
})

describe('Zerodha secrets', () => {
  it('scrubs tokens and secret keys from technical details', () => {
    const clean = scrubTechnical({
      session_state: 'ok',
      access_token: 'leak',
      api_secret: 'leak',
      nested: { refresh_token: 'leak', symbols_ticking: 3 },
    }) as Record<string, unknown>
    expect(clean.access_token).toBeUndefined()
    expect(clean.api_secret).toBeUndefined()
    expect((clean.nested as Record<string, unknown>).refresh_token).toBeUndefined()
    expect((clean.nested as Record<string, unknown>).symbols_ticking).toBe(3)
    expect(hasSecretKey('kite_access_token')).toBe(true)
    expect(technicalLines({ access_token: 'x', heartbeat: '09:16' }).join(' ')).not.toContain('access_token')
  })
})

describe('keyboard and accessibility', () => {
  it('treats Enter and Space as activation keys', () => {
    expect(isActivatingKey('Enter')).toBe(true)
    expect(isActivatingKey(' ')).toBe(true)
    expect(isActivatingKey('Tab')).toBe(false)
    expect(laneAriaLabel('automation', { status: 'Working' })).toMatch(/AUTOMATION: Working/)
  })
})

describe('live-money safety', () => {
  it('rejects any live execution control and keeps money locked', () => {
    expect(isSafeHomeControl('LIVE_BUY')).toBe(false)
    expect(isSafeHomeControl('BROKER_SELL')).toBe(false)
    expect(isSafeHomeControl('UNLOCK_LIVE_MONEY')).toBe(false)
    expect(filterSafeActions([
      { label: 'Buy live', control: 'LIVE_BUY' },
      { label: 'Pause new entries', control: 'PAUSE_NEW_PAPER_ENTRIES' },
    ]).map((a) => a.control)).toEqual(['PAUSE_NEW_PAPER_ENTRIES'])
    FORBIDDEN_HOME_CONTROLS.forEach((name) => expect(isSafeHomeControl(name)).toBe(false))
    expect(liveMoneyStillLocked(true, { live_locked: true })).toBe(true)
    expect(checkSystemRows({
      lanes: [
        { id: 'data', label: 'Data', status: 'Ready' },
        { id: 'live_money', label: 'Live Money', status: 'Locked' },
      ],
    }, {}).some((row) => row.id === 'live_money' && row.status === 'Locked')).toBe(true)
  })
})
