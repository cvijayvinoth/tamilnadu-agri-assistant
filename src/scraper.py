"""Web scraping: single page extraction plus a bounded, polite site crawl.

Safety posture
--------------
* Only http/https URLs are followed.
* Private/loopback/link-local hosts are refused (SSRF guard).
* robots.txt is honoured by default.
* Hard caps on pages, depth, response size and a delay between requests, so a
  crawl of a .gov.in site can never run away.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import urllib.robotparser as robotparser
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse

from config import (
    CRAWL_DELAY_SECONDS,
    CRAWL_MAX_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_TIMEOUT,
    MAX_PAGE_BYTES,
    RESPECT_ROBOTS_TXT,
    USER_AGENT,
)

_HEADERS = {"User-Agent": USER_AGENT}


@dataclass
class PageLink:
    """One discovered URL, before the user approves it for ingestion."""

    url: str
    title: str = ""
    is_pdf: bool = False
    depth: int = 0


@dataclass
class CrawlResult:
    pages: list = field(default_factory=list)   # list[PageLink]
    skipped: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    stopped_reason: str = ""


# ------------------------------------------------------------------ URL rules
def normalize_url(url: str) -> str:
    """Drop fragments and trailing slashes so the visited-set dedupes properly."""
    try:
        p = urlparse(url.strip())
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", p.query, ""))
    except Exception:
        return url.strip()


def is_safe_url(url: str) -> tuple[bool, str]:
    """Block non-web schemes and any host that resolves to a private address."""
    try:
        p = urlparse(url)
    except Exception:
        return False, "Malformed URL."

    if p.scheme not in ("http", "https"):
        return False, "Only http and https URLs are supported."
    if not p.hostname:
        return False, "URL has no host."

    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False, "Refusing to fetch a private or local address."
    except socket.gaierror:
        return False, "That host could not be resolved."
    except ValueError:
        pass

    return True, ""


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


def same_domain(a: str, b: str) -> bool:
    try:
        ha, hb = urlparse(a).hostname or "", urlparse(b).hostname or ""
        return ha.removeprefix("www.") == hb.removeprefix("www.")
    except Exception:
        return False


def _robots_allowed(url: str) -> bool:
    if not RESPECT_ROBOTS_TXT:
        return True
    try:
        p = urlparse(url)
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True          # an unreachable robots.txt is not a prohibition


# ------------------------------------------------------------------ fetching
def fetch(url: str):
    """GET a URL with all the safety checks applied. Returns a requests Response."""
    import requests

    ok, why = is_safe_url(url)
    if not ok:
        raise RuntimeError(why)
    if not _robots_allowed(url):
        raise RuntimeError("robots.txt disallows fetching that URL.")

    resp = requests.get(url, headers=_HEADERS, timeout=CRAWL_TIMEOUT, stream=True)
    resp.raise_for_status()

    declared = int(resp.headers.get("Content-Length") or 0)
    if declared and declared > MAX_PAGE_BYTES:
        raise RuntimeError("That document is too large to ingest safely.")

    body = b""
    for block in resp.iter_content(65536):
        body += block
        if len(body) > MAX_PAGE_BYTES:
            raise RuntimeError("That document is too large to ingest safely.")
    resp._content = body
    return resp


# ----------------------------------------------------------------- discovery
def discover_links(url: str, same_domain_only: bool = True,
                   pdf_only: bool = False) -> list[PageLink]:
    """List the links on a page, so the user can approve before ingesting."""
    from bs4 import BeautifulSoup

    resp = fetch(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    found: dict[str, PageLink] = {}
    for a in soup.find_all("a", href=True):
        raw = urljoin(url, a["href"])
        if not raw.startswith(("http://", "https://")):
            continue
        norm = normalize_url(raw)
        if norm in found:
            continue
        if same_domain_only and not same_domain(url, norm):
            continue
        pdf = is_pdf_url(norm)
        if pdf_only and not pdf:
            continue
        found[norm] = PageLink(
            url=norm,
            title=" ".join(a.get_text(" ", strip=True).split())[:120],
            is_pdf=pdf,
            depth=1,
        )
    return list(found.values())


def crawl_site(start_url: str, max_pages: int = CRAWL_MAX_PAGES,
               max_depth: int = CRAWL_MAX_DEPTH,
               same_domain_only: bool = True) -> CrawlResult:
    """Breadth-first crawl, hard-capped on both pages and depth."""
    from bs4 import BeautifulSoup

    result = CrawlResult()
    start = normalize_url(start_url)

    ok, why = is_safe_url(start)
    if not ok:
        result.errors.append(why)
        return result

    seen = {start}
    queue = deque([(start, 0)])

    while queue:
        if len(result.pages) >= max_pages:
            result.stopped_reason = f"Reached the {max_pages}-page limit."
            break

        url, depth = queue.popleft()
        try:
            resp = fetch(url)
        except Exception as exc:
            result.errors.append(f"{url}: {exc}")
            continue

        ctype = resp.headers.get("Content-Type", "").lower()
        pdf = "pdf" in ctype or is_pdf_url(url)

        title = ""
        if not pdf:
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()[:120]
            except Exception:
                pass
        result.pages.append(PageLink(url=url, title=title, is_pdf=pdf, depth=depth))

        # Only HTML pages expand the frontier; PDFs are leaves.
        if pdf or depth >= max_depth:
            continue
        try:
            for a in BeautifulSoup(resp.text, "html.parser").find_all("a", href=True):
                nxt = normalize_url(urljoin(url, a["href"]))
                if not nxt.startswith(("http://", "https://")) or nxt in seen:
                    continue
                if same_domain_only and not same_domain(start, nxt):
                    continue
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        except Exception:
            pass

        time.sleep(CRAWL_DELAY_SECONDS)     # be a polite guest

    return result
