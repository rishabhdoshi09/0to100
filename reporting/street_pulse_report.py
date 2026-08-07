"""Daily Street Pulse PDF generation — evidence digest, not a signal desk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reporting.research_dossier import DEFAULT_REPORT_DIR, _report_path


def generate_street_pulse_report(
    *,
    force: bool = True,
    report_dir: str | Path | None = None,
) -> Path:
    from reporting.pdf_renderer import render_street_pulse_pdf
    from reports.street_pulse import build_pulse, load_pulse

    pulse = build_pulse(persist=True) if force else (load_pulse() or build_pulse(persist=True))
    target = Path(report_dir or DEFAULT_REPORT_DIR)
    stamp = str(pulse.get("date") or "latest").replace(" ", "_")
    path = _report_path("daily_street_pulse", stamp, target)
    render_street_pulse_pdf(pulse, path)
    return path


def pulse_brief_for_api(force: bool = False) -> dict[str, Any]:
    from reports.street_pulse import pulse_api_payload

    return pulse_api_payload(force=force)
