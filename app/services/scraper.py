from bs4 import BeautifulSoup
import requests

def scrape_wikipedia(url: str) -> str:
     response = requests.get(url)
     soup = BeautifulSoup(response.text, "html.parser")
     paragraphs = [p.get_text() for p in soup.find_all("p")]
     return "\n".join(paragraphs)

if __name__ == "__main__":
     url = "https://en.wikipedia.org/wiki/Data_science"
     content = scrape_wikipedia(url)
     print(content)  # Print first 500 characters# webscraping
from bs4 import BeautifulSoup
import requests

def get_paragraphs(url):
     headers = {'User-Agent': 'DemoScraper/1.0 (kiruthika.sky102@gmail.com)'}
     try:
          response = requests.get(url, headers=headers)
          soup = BeautifulSoup(response.content, 'html.parser')

          # Use the page title if available, otherwise fallback to "Untitled Page"
          title = soup.title.string if soup.title else "Untitled Page"
          
          # Extract paragraphs
          paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
          
          # Build structured data with metadata
          data = []
          for index, paragraph in enumerate(paragraphs):
               data.append({
                    "text": paragraph,
                    "source_url": url,
                    "title": title,
                    "chunk_index": index
               })    
          return data  # returns list of dicts with metadata
     
     except Exception as e:
          print(f"Error fetching {url}: {e}")
          return []