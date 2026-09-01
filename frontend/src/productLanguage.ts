export type DisplayDepth = 'simple' | 'professional'

export type PageGuide = {
  title: string
  purpose: string
  questions: string[]
  action: string
  doesNot: string
}

export const PAGE_GUIDE: Record<string, PageGuide> = {
  Home: {
    title: 'Home',
    purpose: 'One place to see if QuantTerm is running, what it is doing, and whether you need to do anything.',
    questions: [
      'Do I need to do anything?',
      'What is QuantTerm doing now, and what comes next?',
      'Did the paper bot trade — and why or why not?',
      'Is there enough real evidence yet? Is live money still locked?',
    ],
    action: 'Leave it running. If Home shows one button, click that. Do not open a second terminal.',
    doesNot: 'A green market or a sniper tag is not an instruction to buy. Live money stays locked.',
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
  'Market Scanner': {
    title: 'Market Scanner',
    purpose: 'Ranks saved whole-market research by SEPA Best Setups, momentum, breakout or long-term state.',
    questions: [
      'When was the universe last scanned?',
      'How many symbols were processed?',
      'Why did a stock qualify?',
    ],
    action: 'Run a fresh scan once — it fills every tab — then inspect one stock in Stock Intelligence.',
    doesNot: 'A scanner match is a research candidate, not a guaranteed trade.',
  },
  Scanner: {
    title: 'Discover',
    purpose: 'Ranks saved whole-market research by momentum, breakout, setup quality or avoidance state.',
    questions: [
      'When was the universe last scanned?',
      'How many symbols were processed?',
      'Why did a stock qualify?',
      'What invalidates or disqualifies it?',
    ],
    action: 'Run a fresh scan once — it fills every tab — then inspect one stock in Stock Intelligence.',
    doesNot: 'A scanner match is a research candidate, not a guaranteed trade.',
  },
  Recommendations: {
    title: 'Recommendations',
    purpose: 'Mixture of independent experts over QuantTerm evidence. Buy needs two evidence families, a why-now, and a ready entry — not the highest SEPA score.',
    questions: [
      'Which thesis generated this card, and which evidence families agree?',
      'Is this High Conviction, a Good Setup, or Watch / not ready?',
      'Is CMP live or EOD, and how old is the scan?',
      'What is upside from entry vs room to target?',
      'Which tracked picks are still open vs resolved?',
    ],
    action: 'Open a card, read why now, then Investigate for fundamentals vs the setup.',
    doesNot: 'A Buy badge is research classification — not a broker order or return promise.',
  },
  'Market Reports': {
    title: 'Market Reports',
    purpose: 'Daily Market Pulse assembled from live scanners — movers, breadth notes and breakout context.',
    questions: [
      'What changed in the market today?',
      'Which names are buzzing or near breakouts?',
      'Is this report from today IST or an older saved pulse?',
    ],
    action: 'Read takeaways, then jump to Recommendations or Scanner for names mentioned.',
    doesNot: 'A pulse summary is context, not a trade ticket.',
  },
  'Stock Intelligence': {
    title: 'Stock Intelligence',
    purpose: 'Combines price structure, technicals, current fundamentals, risks, news and source dates. Investigate is the second-stage due-diligence view — it is not a new scanner.',
    questions: [
      'What is the current trend and volatility?',
      'Are fundamentals complete and fresh?',
      'Does the sector-framework evidence support, leave unchanged, or contradict the technical setup?',
      'Which missing fact could change the conclusion?',
    ],
    action: 'Open Investigate on a shortlisted name, then refresh missing fundamentals in Research Data if coverage is thin.',
    doesNot: 'Investigate is not a new scanner and does not place or recommend orders. Empty stays empty.',
  },
  'Stock Investigator': {
    title: 'Stock Investigator',
    purpose: 'Manually look up any supported NSE name. Typing ICICI should suggest ICICIBANK — ICICI Bank. The same StockResearchEngine as scanner Investigate runs.',
    questions: [
      'What exactly am I buying?',
      'Are fundamentals improving or deteriorating?',
      'Are there hidden financial or governance risks?',
      'Do the fundamentals support or weaken a technical setup if one exists?',
    ],
    action: 'Type a ticker or company name, pick the match, then Acquire if files are missing.',
    doesNot: 'This is not a scanner, not a buy list, and never uses a language model for scoring or verdicts.',
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
    action: 'Scan market once. Refresh funds only if Screener snapshots are stale. Then inspect the company dossier.',
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
  Education: {
    title: 'Education',
    purpose: 'Crunches curated market news into learnable micro/macro/policy/F&O cards plus evergreen concepts — without inventing articles.',
    questions: [
      'Is this macro weather or company micro?',
      'What is the teach point, not just the headline?',
      'Is there an original source URL to verify?',
      'Am I treating education as context instead of a trade tip?',
    ],
    action: 'Read the teach point, open the source when present, then review linked symbols in Stock Intelligence.',
    doesNot: 'Education never invents blogs and never places or recommends orders.',
  },
  Backtest: {
    title: 'Backtests',
    purpose: 'Show only backtests connected to the live recommendation ensemble. Unproven hash stays UNVERIFIED.',
    questions: [
      'Which strategy_id and rules_hash produced today’s signal?',
      'Is parity proven or UNVERIFIED?',
      'Are paper diary rows being mistaken for ensemble performance?',
    ],
    action: 'Read the parity line first. Then open Learning for settled paper and scan rejects.',
    doesNot: 'A related scanner calibration or paper StrategySpec is not this ensemble’s track record.',
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
    doesNot: 'This is not a directional F&O signal desk and does not calculate Greeks. An Acquire snapshot can show nearest-expiry OI, IV, PCR and max pain on Stock Intelligence / Investigate.',
  },
  'Market Internals': {
    title: 'Market & Breadth',
    purpose: 'Explains the environment in which stock signals are being interpreted.',
    questions: ['Is breadth supportive?', 'Which sectors lead or lag?', 'Is volatility expanding?', 'Does the regime support new risk?'],
    action: 'Use the market state as a portfolio constraint, not as a stock recommendation.',
    doesNot: 'A healthy index does not make every individual setup safe.',
  },
  'Paper Portfolio': {
    title: 'My Holdings',
    purpose: 'Records demat holdings plus simulated positions, risk and outcomes so the system can be judged honestly.',
    questions: ['How much open risk exists?', 'Are exits protected?', 'How many closed trades exist?', 'Is the sample large enough for performance statistics?'],
    action: 'Review position-level risk and refusals before looking at headline P&L.',
    doesNot: 'Paper equity is evidence plumbing, not the brain or heart of QuantTerm.',
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
  'System Health': {
    title: 'System Health',
    purpose: 'Independent lanes for auth, universe, OHLCV, news, scan, workers and API contract. No single green light.',
    questions: ['Which lane is stale or broken?', 'Is the operations worker alive?', 'Is scan coverage a coverage problem or an auth problem?'],
    action: 'Fix the named lane. Do not treat a live PID as product-ready.',
    doesNot: 'A healthy worker does not make missing fundamentals healthy.',
  },
  Strategies: {
    title: 'Strategies',
    purpose: 'Names the live recommendation ensemble and each method with id, version and rules_hash.',
    questions: ['What is in production?', 'What is research-only?', 'Is backtest parity proven?'],
    action: 'Open a recommendation and confirm it cites QT_RECO_ENSEMBLE, not a paper strategy.',
    doesNot: 'Research-only StrategySpec rows never rank today’s BUY list.',
  },
  Learning: {
    title: 'Learning / Decision Journal',
    purpose: 'Shows measurable sample sizes, settled paper, skipped names and scan rejects.',
    questions: ['How many settled observations exist?', 'What was learned, with numbers?', 'Which names were checked and rejected?'],
    action: 'Read headlines first. Empty sample means no performance claim.',
    doesNot: 'QuantTerm never writes “AI is learning” without a measured artifact.',
  },
  Coverage: {
    title: 'Coverage',
    purpose: 'Separates requested, checked, qualified, no-setup, excluded and failed names from the last scan.',
    questions: ['How large was the requested universe?', 'Which names failed or were missing?', 'Was this ticker scanned?'],
    action: 'Inspect a ticker that did not qualify.',
    doesNot: 'Coverage is not a quality score and not a buy list.',
  },
  'Long-Term Picks': {
    title: 'Long-Term Research',
    purpose: 'Filters technically eligible companies using current quality, growth, leverage and valuation evidence.',
    questions: [
      'Is fundamental coverage adequate?',
      'Is the company quality-backed or only technically strong?',
      'Is price timing favourable or extended?',
      'Which risk flag needs manual review?',
    ],
    action: 'Scan market once. Refresh funds only if Screener snapshots are stale. Then inspect the company dossier.',
    doesNot: 'A long-term classification is not a promise of compounding or future returns.',
  },
  'Market Overview': {
    title: 'Market & Breadth',
    purpose: 'Explains the environment in which stock signals are being interpreted.',
    questions: ['Is breadth supportive?', 'Which sectors lead or lag?', 'Is volatility expanding?', 'Does the regime support new risk?'],
    action: 'Use the market state as a portfolio constraint, not as a stock recommendation.',
    doesNot: 'A healthy index does not make every individual setup safe.',
  },
}

export const GLOSSARY: Record<string, string> = {
  Candidate: 'A stock worth reviewing; not an instruction to trade.',
  'Setup Quality': 'A 0–100 checklist of how complete this setup looks (score, regime, volume, RSI). It is not the chance the trade works. Observed hit-rate lives on Evidence / n.',
  Conviction: 'Legacy name for Setup Quality — a weighted checklist, not a win probability.',
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
