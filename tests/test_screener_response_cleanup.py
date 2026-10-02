from __future__ import annotations

import fundamentals.screener_deep as SD


class _FakeResponse:
    def __init__(self, status_code=200, text="<html></html>", url="https://www.screener.in/"):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.closed = False

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, gets=None, post=None):
        self.gets = list(gets or [])
        self.post_response = post
        self.seen_gets = []
        self.seen_posts = []

    def get(self, url, **kwargs):
        self.seen_gets.append((url, kwargs))
        if not self.gets:
            raise AssertionError("unexpected GET")
        return self.gets.pop(0)

    def post(self, url, **kwargs):
        self.seen_posts.append((url, kwargs))
        if self.post_response is None:
            raise AssertionError("unexpected POST")
        return self.post_response


def _fetcher(session, *, warmed=True):
    fetcher = SD.ScreenerDeepFetcher.__new__(SD.ScreenerDeepFetcher)
    fetcher._session = session
    fetcher._warmed = warmed
    return fetcher


def test_warm_session_closes_homepage_response(monkeypatch):
    response = _FakeResponse(status_code=200, text="<html></html>")
    session = _FakeSession(gets=[response])
    fetcher = _fetcher(session, warmed=False)

    monkeypatch.setattr(SD.time, "sleep", lambda *_: None)
    monkeypatch.setattr(SD.settings, "screener_email", "")
    monkeypatch.setattr(SD.settings, "screener_password", "")

    fetcher._warm_session()

    assert response.closed is True
    assert fetcher._warmed is True


def test_login_closes_post_response(monkeypatch):
    response = _FakeResponse(status_code=200, url="https://www.screener.in/dashboard/")
    session = _FakeSession(post=response)
    fetcher = _fetcher(session)

    monkeypatch.setattr(SD.time, "sleep", lambda *_: None)

    fetcher._login("user@example.test", "secret", "<input name='csrfmiddlewaretoken' value='abc'>")

    assert response.closed is True


def test_fetch_page_closes_primary_response(monkeypatch):
    response = _FakeResponse(
        status_code=200,
        text="<html><body><div class='about'>hello</div></body></html>",
    )
    session = _FakeSession(gets=[response])
    fetcher = _fetcher(session)

    monkeypatch.setattr(SD.time, "sleep", lambda *_: None)

    url, soup = fetcher._fetch_page("INFY")

    assert "INFY" in url
    assert soup.find("div", {"class": "about"}).get_text(strip=True) == "hello"
    assert response.closed is True


def test_fetch_page_closes_404_before_fallback_and_closes_fallback(monkeypatch):
    first = _FakeResponse(status_code=404, text="not found")
    second = _FakeResponse(
        status_code=200,
        text="<html><body><div class='about'>fallback</div></body></html>",
    )
    session = _FakeSession(gets=[first, second])
    fetcher = _fetcher(session)

    monkeypatch.setattr(SD.time, "sleep", lambda *_: None)

    url, soup = fetcher._fetch_page("INFY")

    assert url.endswith("/company/INFY/")
    assert soup.find("div", {"class": "about"}).get_text(strip=True) == "fallback"
    assert first.closed is True
    assert second.closed is True
