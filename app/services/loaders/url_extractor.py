"""
url_loader.py
----------------
Responsible for extracting clean text from any webpage URL.
Returns unified structured documents.
"""

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "KeerthiScraper/1.0 (+https://github.com/keerthi)"
}


def process_url(url: str):
    """
    Extract clean text from a webpage.
    Always returns a list of standardized document dicts.
    """

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"URL fetch failed for {url}: {e}")

    try:
        soup = BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        raise RuntimeError(f"HTML parsing failed for {url}: {e}")

    title = soup.title.string.strip() if soup.title else "Untitled"

    # Collect all paragraph text
    paragraphs = [
        p.get_text().strip()
        for p in soup.find_all("p")
        if p.get_text().strip()
    ]

    if not paragraphs:
        # fallback: some sites use <div> for article body
        paragraphs = [
            div.get_text().strip()
            for div in soup.find_all("div")
            if div.get_text().strip()
        ]

    output = []
    for i, text in enumerate(paragraphs):
        output.append(
            {
                "text": text,
                "source": url,
                "type": "url",
                "title": title,
                "chunk_index": i
            }
        )

    return output
