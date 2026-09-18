from research.auto_research.paper_book import PaperBook


def test_paper_position_sector_survives_snapshot_restore():
    book = PaperBook()
    pos = book.open_position(
        "QT_RECO",
        "INFY",
        100.0,
        95.0,
        115.0,
        "2026-09-18",
        20,
        risk_pct_of_capital=1.0,
    )
    assert pos is not None
    pos.sector = "IT"

    restored = PaperBook()
    restored.restore(book.snapshot())

    assert len(restored.open) == 1
    saved = next(iter(restored.open.values()))
    assert saved.symbol == "INFY"
    assert saved.sector == "IT"
