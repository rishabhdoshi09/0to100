export type DisplayDepth = 'simple' | 'professional'

export type PageGuide = {
  title: string
  purpose: string
  questions: string[]
  action: string
  doesNot: string
}

export const PAGE_GUIDE: Record<string, PageGuide> = {
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
  'Confirmed Breakouts': {
    title: 'Confirmed Breakouts',
    purpose: 'Collects live sniper pivot confirms into one durable board, then ranks only that list.',
    questions: [
      'Which names actually confirmed a held breakout today?',
      'Among those, which still look strong on momentum and measured edge?',
      'Do fundamentals support a longer-horizon shortlist?',
      'What evidence is still missing before considering a buy tomorrow?',
    ],
    action: 'Wait for sniper confirms, click Evaluate board, then open Stock Intelligence on PRIORITY / CANDIDATE names.',
    doesNot: 'A sniper confirm or Priority verdict is not a live buy order and never invents missing edge or fundamentals.',
  },
  'Stock Intelligence': {
    title: 'Stock Intelligence',
    purpose: 'Combines price structure, technicals, current fundamentals, growth/outlook (evidence-only), risks, news and source dates.',
    questions: [
      'What is the current trend and volatility?',
      'Are sales/profit growth and margins visible in verified data?',
      'Is there uploaded concall / management guidance, or is it still missing?',
      'Which missing fact could change the conclusion?',
    ],
    action: 'Open Outlook after fundamentals refresh. Upload concall/guidance under Research Data when you have filings — never invent quotes.',
    doesNot: 'Outlook never invents FY targets or management quotes. Current fundamentals are not point-in-time historical evidence unless labelled.',
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
  'Daily Pulse': {
    title: 'Daily Pulse',
    purpose: 'A simple daily market digest from your scan, prices, sectors, options mood, and news — plus one-tap Telegram send.',
    questions: [
      'What moved today in plain English?',
      'Which sectors are hot or cold?',
      'Is options mood friendly, mixed, or hostile for new longs?',
      'What is buzzing, gaining strength, or losing steam?',
      'What should I watch tomorrow?',
    ],
    action: 'Rebuild after a market scan, tap Send to Telegram, then open a stock in Stock Intelligence before any paper trade.',
    doesNot: 'Not a buy desk. Labels are research context only — missing sections stay missing.',
  },
  'Active Buys': {
    title: 'Active Buys',
    purpose: 'Stock results for names you are buying — entry → now %, 1D/5D move, optional qty estimate — plus technical health if averages or support break.',
    questions: [
      'Am I up or down vs my entry?',
      'What did this stock do today and over 5 sessions?',
      'Is price still above the 20/50/200-day averages?',
      'Did it break 20-session or 60-session swing support?',
      'Is this a research warning, not an automatic sell?',
    ],
    action: 'Add symbol + entry (qty optional for ₹ estimate). Refresh results during market hours. Use My Holdings for demat P&L truth.',
    doesNot: 'Not a broker order desk. Qty P&L is your estimate, not demat. Warnings do not sell for you. Missing entry stays missing.',
  },
  'US Market': {
    title: 'US Market',
    purpose: 'Retail US plane — NASDAQ Trader listings, Yahoo EOD cache, liquid scan scope, paper autopilot only.',
    questions: [
      'Is US history cache ready?',
      'Which scope was scanned (S&P 500 default)?',
      'Are quotes delayed free-feed prints?',
      'Am I treating paper autopilot as live?',
    ],
    action: 'Prepare US history, run US scan, open setups in US Stock.',
    doesNot: 'No live US broker orders and no US options desk.',
  },
  'US Scanner': {
    title: 'US Scanner',
    purpose: 'Same setup engine as NSE, pointed at US equities with S&P relative strength and a liquid-name quality floor.',
    questions: ['Is the setup above the $5 / turnover floor?', 'What is entry/stop/target?', 'Is chase risk flagged?'],
    action: 'Open the ticker in US Stock and verify the Yahoo daily chart.',
    doesNot: 'Scanner output is not a live US order ticket.',
  },
  'US Stock': {
    title: 'US Stock',
    purpose: 'Ticker workspace with Yahoo daily history and last scan setup — fundamentals/options marked unavailable when missing.',
    questions: ['Do daily bars exist?', 'Is there a scan row?', 'What exactly is unavailable?'],
    action: 'Use chart + setup as context only; prepare history if bars are empty.',
    doesNot: 'Does not invent US fundamentals, Greeks, or broker fills.',
  },
  'F&O Desk': {
    title: 'F&O Coverage',
    purpose: 'Maps futures eligibility, loads live OI/IV/PCR/max pain, and builds a research positioning stance (not a buy ticket).',
    questions: [
      'Is the cash symbol mapped to the current instrument master?',
      'Is options positioning SUPPORTIVE, CAUTION, or HOSTILE for a long watch?',
      'What OI walls / max-pain gravity sit near spot?',
      'Does cash-scan evidence agree, or is chase risk flagged?',
    ],
    action: 'Refresh the live chain, read the positioning stance, then open Stock Intelligence / pre-trade before any paper rehearsal.',
    doesNot: 'SUPPORTIVE is not BUY. No Greeks, no live orders, no invented fills — paper-first research only.',
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
