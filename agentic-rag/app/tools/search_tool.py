import requests
from bs4 import BeautifulSoup


def simple_web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    url = "https://duckduckgo.com/html/"
    response = requests.post(url, data={"q": query}, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for item in soup.select(".result")[:max_results]:
        title = item.select_one(".result__title")
        link = item.select_one(".result__a")
        snippet = item.select_one(".result__snippet")
        if title and link:
            results.append(
                {
                    "title": title.get_text(" ", strip=True),
                    "url": link.get("href", ""),
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                }
            )
    return results

