import re
import json
import html
import requests

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import xml.etree.ElementTree as ET

@tool
def add_numbers(text: str) -> str:
    """
    Add two numbers.

    Input may look like:
    - 3, 5
    - '3, 5'
    - add 3 and 5
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    if len(numbers) < 2:
        return "Please provide two numbers, like: 3, 5"

    a = float(numbers[0])
    b = float(numbers[1])

    result = a + b

    if result.is_integer():
        return str(int(result))

    return str(result)


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the given text.
    """
    return text[::-1]

##################################
# Wikipedia tool
##################################
def html_to_readable_text(value: str) -> str:
    """Remove HTML tags/entities from a Wikipedia snippet."""
    if not value:
        return ""

    text = re.sub(r"<[^>]+>", "", value)
    return html.unescape(text).strip()


def wikipedia_data_to_readable_text(data: dict) -> str:
    """Convert Wikipedia search API response into readable text."""
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

        lines.append(f"{index}. {title}")
        lines.append(f"Summary: {snippet}")
        lines.append("")

    return "\n".join(lines).strip()

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a topic and return readable search results.
    Use this for factual, encyclopedic, medical, scientific, historical, or biographical lookup.
    """
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
            },
            headers={
                "User-Agent": "llm-service/0.1"
            },
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

        return wikipedia_data_to_readable_text(data)

    except Exception as e:
        return f"Wikipedia search failed for '{query}': {type(e).__name__}: {e}"
    


##################################
# Wikipedia tool
##################################
@tool
def arxiv_search(query: str) -> str:
    """
    Search arXiv for academic papers relevant to the user's question.
    Returns paper metadata only: title, authors, abstract, arXiv URL, PDF URL,
    DOI when available, publication date, and categories.
    Does not download or parse PDFs.
    """
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 5,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers={
                "User-Agent": "llm-service/0.1"
            },
            timeout=15,
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"No arXiv papers found for: {query}"

        papers = []

        for index, entry in enumerate(entries, start=1):
            title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
            summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
            published = entry.findtext("atom:published", default="", namespaces=ns)
            updated = entry.findtext("atom:updated", default="", namespaces=ns)

            authors = [
                author.findtext("atom:name", default="", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]

            categories = [
                category.attrib.get("term", "")
                for category in entry.findall("atom:category", ns)
            ]

            abs_url = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = abs_url.rsplit("/", 1)[-1] if abs_url else ""

            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break

            doi = entry.findtext("arxiv:doi", default="", namespaces=ns)

            papers.append(
                "\n".join(
                    [
                        f"{index}. {title}",
                        f"arXiv ID: {arxiv_id}",
                        f"Authors: {', '.join(authors)}",
                        f"Published: {published}",
                        f"Updated: {updated}",
                        f"Categories: {', '.join(categories)}",
                        f"DOI: {doi or 'Not available'}",
                        f"Abstract URL: {abs_url}",
                        f"PDF URL: {pdf_url or 'Not available'}",
                        f"Summary: {summary}",
                    ]
                )
            )

        return "\n\n".join(papers)

    except Exception as e:
        return f"arXiv search failed for '{query}': {type(e).__name__}: {e}"


TOOLS = [
    add_numbers,
    reverse_text,
    wikipedia_search,
    arxiv_search
]