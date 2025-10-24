import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def crawl_website(url):
    print(f"\nCrawling URL: {url}\n")

    try:
        # Step 1: Fetch the page
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        html = response.text

        # Step 2: Parse the HTML
        soup = BeautifulSoup(html, "html.parser")

        # Step 3: Extract page title
        page_title = soup.title.string.strip() if soup.title else "No title found"
        print(f"Page Title: {page_title}\n")

        # Step 4: Extract all hyperlinks
        print("=== Extracted Hyperlinks ===")
        links = []
        for link in soup.find_all("a", href=True):
            full_link = urljoin(url, link["href"])
            if urlparse(full_link).scheme in ["http", "https"]:
                links.append(full_link)

        for i, link in enumerate(links[:15], 1):  # show first 15
            print(f"{i}. {link}")

        # Step 5: Try to extract meaningful content (headings, prices, etc.)
        print("\n=== Extracted Text Content (Sample) ===")
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

        print("\n-- Headings --")
        for i, h in enumerate(headings[:5], 1):
            print(f"{i}. {h}")

        print("\n-- Sample Paragraphs --")
        for i, p in enumerate(paragraphs[:3], 1):
            print(f"{i}. {p[:200]}...")  # show first 200 chars

    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")


if __name__ == "__main__":
    print("=== Generic Web Crawler ===")
    user_url = input("Enter any website URL (e.g., https://example.com): ").strip()
    if not user_url.startswith("http"):
        user_url = "https://" + user_url
    crawl_website(user_url)
# websites to use :
# https://books.toscrape.com/
# https://quotes.toscrape.com/
# https://httpbin.org/html

