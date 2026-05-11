"""
Earnings Call Analyst Agent.

Ingests a YouTube earnings call URL, extracts the full transcript via
youtube-transcript-api (no audio download needed — uses auto-generated
or manual captions), then sends to DeepSeek R1 for structured extraction.

DeepSeek R1 (deepseek-reasoner) is used for its chain-of-thought reasoning —
it double-checks numbers, detects management tone shifts, and cross-references
statements within the transcript before outputting the JSON report.

Hardware note: all processing is API-based; no local model inference.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from ai.deepseek_dual import DeepSeekDual

_ANALYSIS_PROMPT = """You are a sell-side equity research analyst specialising in Indian listed companies.

Given the earnings call transcript below, extract a structured investment-grade analysis.

Be precise and evidence-based — quote directly from the transcript where possible.
If a data point is not mentioned, use null.

Return ONLY a JSON object with these keys:

{
  "revenue_beat": true | false | null,
  "revenue_actual": "<value with units, e.g. ₹12,340 Cr>",
  "revenue_estimate": "<analyst consensus if mentioned>",
  "eps_beat": true | false | null,
  "eps_actual": "<value>",
  "eps_estimate": "<value>",
  "guidance_change": "RAISED" | "LOWERED" | "MAINTAINED" | "WITHDRAWN" | "NOT_PROVIDED",
  "guidance_detail": "<brief description of what guidance was given>",
  "key_quotes": [
    "<most impactful management quote 1>",
    "<most impactful management quote 2>",
    "<most impactful analyst Q&A exchange (paraphrased)>"
  ],
  "management_tone": "OPTIMISTIC" | "CAUTIOUS" | "DEFENSIVE" | "NEUTRAL",
  "analyst_qa_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "red_flags": ["<concern 1 if any>"],
  "catalysts": ["<positive catalyst 1 if any>"],
  "outlook": "<2-3 sentence forward-looking summary>",
  "overall_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL"
}"""


class EarningsAgent:
    def __init__(self) -> None:
        self.llm = DeepSeekDual()

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze(self, youtube_url: str, company_name: str) -> Dict[str, Any]:
        """
        Fetch transcript from YouTube URL and return structured earnings analysis.
        Raises ValueError for invalid URLs or unavailable transcripts.
        """
        video_id = self._extract_video_id(youtube_url)
        transcript = self._fetch_transcript(video_id)
        analysis = self._run_analysis(transcript, company_name)
        analysis["company"] = company_name
        analysis["video_id"] = video_id
        analysis["url"] = youtube_url
        analysis["transcript_chars"] = len(transcript)
        return analysis

    def get_transcript_preview(self, youtube_url: str, chars: int = 2000) -> str:
        """Return the first `chars` of the transcript for a quick sanity check."""
        video_id = self._extract_video_id(youtube_url)
        transcript = self._fetch_transcript(video_id)
        return transcript[:chars]

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_video_id(url: str) -> str:
        patterns = [
            r"(?:v=)([0-9A-Za-z_-]{11})",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
            r"(?:shorts/)([0-9A-Za-z_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(
            f"Cannot extract video ID from URL: {url}\n"
            "Expected format: https://www.youtube.com/watch?v=XXXXXXXXXXX"
        )

    @staticmethod
    def _fetch_transcript(video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
        except ImportError:
            raise ImportError(
                "youtube-transcript-api not installed. Run: pip install youtube-transcript-api"
            )

        try:
            # Try English first, then any available language
            try:
                segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-IN"])
            except NoTranscriptFound:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_generated_transcript(
                    transcript_list._manually_created_transcripts or
                    list(transcript_list._generated_transcripts.keys())
                )
                segments = transcript.fetch()
        except TranscriptsDisabled:
            raise ValueError(
                f"Transcripts are disabled for video {video_id}. "
                "Try a different earnings call video."
            )

        full_text = " ".join(seg["text"] for seg in segments)

        # DeepSeek R1 context limit: keep last ~30k chars (most recent = most
        # relevant for guidance and Q&A sections)
        if len(full_text) > 30_000:
            full_text = full_text[-30_000:]

        return full_text

    def _run_analysis(self, transcript: str, company_name: str) -> Dict[str, Any]:
        prompt = (
            f"{_ANALYSIS_PROMPT}\n\n"
            f"Company: {company_name}\n\n"
            f"Transcript:\n{transcript}"
        )
        # Use R1 (reasoning=True) — slower but far more accurate for
        # numerical extraction and tone analysis from long transcripts
        result = self.llm.structured_response(prompt, reasoning=True)
        return result
