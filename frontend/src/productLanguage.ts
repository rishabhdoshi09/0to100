export type DisplayDepth = 'simple' | 'professional'

export type PageGuide = {
  title: string
  purpose: string
  questions: string[]
  action: string
  doesNot: string
}

export const PAGE_GUIDE: Record<string, PageGuide> = {
  Today: {
    title: 'Today',
    purpose: 'Best Setups and the scanner watchlist from the last official scan.',
    questions: [
      'Which names cleared a setup today?',
      'Is market data ready?',
      'What did the bot learn from paper?',
    ],
    action: 'Open a setup card, then Paper Desk or Backtest after a paper loss.',
    doesNot: 'A qualify is research, not a broker order.',
  },
  Setups: {
    title: 'Setups',
    purpose: 'Breakouts, Momentum and Long-term lists. Four jobs — do not mix them.',
    questions: ['When did the scan run?', 'Why did this name qualify?'],
    action: 'Filter the list, then inspect one name.',
    doesNot: 'A scanner match is not a guaranteed trade.',
  },
  'Paper Desk': {
    title: 'Paper Desk',
    purpose: 'Simulated book. The bot learns from closed trades every day.',
    questions: ['What is open?', 'What is on cooldown?', 'Is live still locked?'],
    action: 'Enable paper auto from autonomy, then review closed trades.',
    doesNot: 'This page never places a broker order.',
  },
  Backtest: {
    title: 'Backtest',
    purpose: 'Inspect a paper-loss style on past data after costs.',
    questions: ['Did this style pay historically?'],
    action: 'Open a closed paper loss, then review Stock Intelligence.',
    doesNot: 'A backtest does not change today’s BUY list or paper autopilot.',
  },
  Desk: {
    title: 'Desk',
    purpose: 'Market, news, data, stock workspace and system health.',
    questions: ['Is autonomy online?', 'Is history fresh?'],
    action: 'Use the inner tabs. Daily trading lives on Today and Paper Desk.',
    doesNot: 'Desk tools do not unlock live orders.',
  },
  'Command Center': {
    title: 'Home',
    purpose: 'Shows whether QuantTerm has usable data and what deserves attention now.',
    questions: [
      'Is the market-data pipeline fresh?',
      'Did the current scan actually finish?',
      'Which names qualify for technical or long-term review?',
      'What should be refreshed before trusting the screen?',
    ],
    action: 'Start the missing data lane or open one stock for evidence review.',
    doesNot: 'A green market or high score is not an instruction to buy.',
  },
  Scanner: {
    title: 'Discover',
    purpose: 'Ranks saved whole-market research by momentum, breakout, conviction or avoidance state.',
    questions: [
      'When was the universe last scanned?',
      'How many symbols were processed?',
      'Why did a stock qualify?',
      'What invalidates or disqualifies it?',
    ],
    action: 'Run a fresh scan, filter the result, then inspect one stock in Stock Intelligence.',
    doesNot: 'A scanner match is a research candidate, not a guaranteed trade.',
  },
  'Stock Intelligence': {
    title: 'Stock Intelligence',
    purpose: 'Combines price structure, technicals, current fundamentals, risks, news and source dates.',
    questions: [
      'What is the current trend and volatility?',
      'Are fundamentals complete and fresh?',
      'Which evidence supports the shortlist?',
      'Which missing fact could change the conclusion?',
    ],
    action: 'Refresh missing fundamentals or complete the source pack in Research Data.',
    doesNot: 'Current fundamentals are not point-in-time historical evidence unless explicitly labelled.',
  },
  'Long-Term': {
    title: 'Long-Term Research',
    purpose: 'Filters technically eligible companies using current quality, growth, leverage and valuation evidence.',
    questions: [
      'Is fundamental coverage adequate?',
      'Is the company quality-backed or only technically strong?',
      'Is price timing favourable or extended?',
      'Which risk flag needs manual review?',
    ],
    action: 'Run the long-term refresh, filter by quality and coverage, then inspect the company dossier.',
    doesNot: 'A long-term classification is not a promise of compounding or future returns.',
  },
  'Research Data': {
    title: 'Research Data',
    purpose: 'Shows exactly which datasets are fresh, stale, missing, uploaded or awaiting extraction.',
    questions: [
      'What period does each number represent?',
      'Which official source supports it?',
      'Can QuantTerm fetch it automatically?',
      'Is an uploaded document structured and usable or only attached?',
    ],
    action: 'Use the official link, template and upload flow to close one evidence gap.',
    doesNot: 'Attaching a PDF does not automatically make its claims available to the model.',
  },
  'News & Events': {
    title: 'News & Events',
    purpose: 'Adds dated company, regulatory and macro context with source-by-source health.',
    questions: [
      'Is the item official or editorial?',
      'When was it published and fetched?',
      'Which company or derivative is actually linked?',
      'Why might it matter relative to company size or fundamentals?',
    ],
    action: 'Open the original source and treat the item as context for the stock thesis.',
    doesNot: 'News is never a standalone order signal.',
  },
  'F&O Desk': {
    title: 'F&O Coverage',
    purpose: 'Shows current derivative eligibility, nearest futures contract, expiry and lot size.',
    questions: [
      'Is the cash symbol mapped to the current instrument master?',
      'Which expiry and lot size apply?',
      'Which underlyings were excluded and why?',
    ],
    action: 'Refresh the instrument master and inspect mapping exclusions.',
    doesNot: 'This is not yet an option-chain, OI, IV, Greeks or directional strategy engine.',
  },
  'Market Internals': {
    title: 'Market & Breadth',
    purpose: 'Explains the environment in which stock signals are being interpreted.',
    questions: ['Is breadth supportive?', 'Which sectors lead or lag?', 'Is volatility expanding?', 'Does the regime support new risk?'],
    action: 'Use the market state as a portfolio constraint, not as a stock recommendation.',
    doesNot: 'A healthy index does not make every individual setup safe.',
  },
  Portfolio: {
    title: 'Paper Portfolio',
    purpose: 'Records simulated positions, risk and outcomes so the system can be judged honestly.',
    questions: ['How much open risk exists?', 'Are exits protected?', 'How many closed trades exist?', 'Is the sample large enough for performance statistics?'],
    action: 'Review position-level risk and refusals before looking at headline P&L.',
    doesNot: 'Paper equity is evidence plumbing, not the brain or heart of QuantTerm.',
  },
  Automation: {
    title: 'System Health',
    purpose: 'Shows worker liveness, job throughput, failures, retries and owner controls.',
    questions: ['Is the process alive?', 'Is usable data flowing?', 'Which job is active?', 'What exact blocker requires action?'],
    action: 'Resolve the oldest critical blocker or restart only the failed service.',
    doesNot: 'A live PID alone does not mean the product is ready.',
  },
}

export const GLOSSARY: Record<string, string> = {
  Candidate: 'A stock worth reviewing; not an instruction to trade.',
  Conviction: 'A combined evidence score, not certainty.',
  Expectancy: 'Average result per trade after the stated assumptions.',
  Drawdown: 'The fall from a previous portfolio high.',
  Slippage: 'Difference between the expected and actual fill price.',
  'Point-in-time safe': 'Uses only information that was available at that historical moment.',
  'Fundamental coverage': 'How much of the required quality data is actually present.',
  'Chase risk': 'Price is extended enough that a fresh entry may have poor risk/reward.',
  'Market regime': 'The broad trend, breadth and volatility environment.',
  'Source date': 'The period or timestamp the displayed fact actually represents.',
}

export const depthLabel = (depth: DisplayDepth) => depth === 'simple' ? 'Simple' : 'Professional'
