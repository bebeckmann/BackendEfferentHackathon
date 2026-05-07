from langchain_community.utilities import WikipediaAPIWrapper

"""
wikipedia = WikipediaAPIWrapper()

print(wikipedia.run("Lactate Blood"))
"""

import requests
import html
from bs4 import BeautifulSoup
import json
import re


def html_to_readable_text(value: str) -> str:
    """Remove HTML tags/entities from a Wikipedia snippet."""
    if not value:
        return ""

    text = re.sub(r"<[^>]+>", "", value)
    return html.unescape(text).strip()


def wikipedia_response_to_readable_text(response_text: str) -> str:
    """Convert a Wikipedia search JSON response into readable text."""
    data = json.loads(response_text)

    search_info = data.get("query", {}).get("searchinfo", {})
    suggestion = search_info.get("suggestion")
    results = data.get("query", {}).get("search", [])

    lines = []

    if suggestion:
        lines.append(f"Suggested search: {suggestion}")
        lines.append("")

    if not results:
        return "No Wikipedia results found."

    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        snippet = html_to_readable_text(result.get("snippet", ""))
        pageid = result.get("pageid")
        wordcount = result.get("wordcount")

        lines.append(f"{index}. {title}")

        if snippet:
            lines.append(f"   Summary: {snippet}")

        details = []
        if pageid is not None:
            details.append(f"page id: {pageid}")
        if wordcount is not None:
            details.append(f"word count: {wordcount}")

        if details:
            lines.append(f"   Details: {', '.join(details)}")

        lines.append("")

    return "\n".join(lines).strip()


params = {
    "action": "query",
    "list": "search",
    "srsearch": "What is the relationship between initial lactate level and 28-day mortality in septic shock?",
    "format": "json",
}

response = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params=params,
    headers={"User-Agent": "sepsis-atlas/0.1"},
    timeout=10,
)

readable = wikipedia_response_to_readable_text(response.text)
print(readable)