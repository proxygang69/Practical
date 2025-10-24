import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv

def fetch_page(url):
    """Fetch a page safely and return BeautifulSoup object."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching {url}: {e}")
        return None


def extract_page_data(url):
    """Extract title, links, and text content from a single page."""
    soup = fetch_page(url)
    if not soup:
        return None

    title = soup.title.string.strip() if soup.title else "No Title"
    links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True)]
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

    return {
        "url": url,
        "title": title,
        "links": links,
        "headings": headings[:5],
        "paragraphs": paragraphs[:3]
    }


def save_to_csv(data_list, filename="crawl_output.csv"):
    """Save extracted data to CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Page Title", "Headings", "Sample Paragraphs", "Links"])
        for data in data_list:
            writer.writerow([
                data["url"],
                data["title"],
                "; ".join(data["headings"]),
                " ".join(data["paragraphs"]),
                "; ".join(data["links"][:5])  # Save first 5 links
            ])
    print(f"\n✅ Data saved to '{filename}' successfully!")


def crawl_site(start_url, max_pages=3):
    """Basic multi-page crawler with limited depth."""
    visited = set()
    to_visit = [start_url]
    crawled_data = []

    print(f"\nStarting crawl at: {start_url}")
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        print(f"\n🔎 Crawling ({len(visited)+1}/{max_pages}): {current_url}")
        data = extract_page_data(current_url)
        if not data:
            continue

        crawled_data.append(data)
        visited.add(current_url)

        # Add same-domain links for limited crawl
        domain = urlparse(start_url).netloc
        for link in data["links"]:
            if domain in link and link not in visited and len(to_visit) < max_pages:
                to_visit.append(link)

    save_to_csv(crawled_data)


if __name__ == "__main__":
    print("=== Smart Web Crawler ===")
    user_url = input("Enter a website URL (e.g., https://example.com): ").strip()
    if not user_url.startswith("http"):
        user_url = "https://" + user_url

    crawl_site(user_url, max_pages=3)
