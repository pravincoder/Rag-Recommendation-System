import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging
import urllib.parse
from sentence_transformers import SentenceTransformer
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            logging.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None

    def get_page_content(self, start: int, type_num: int) -> str:
        """Fetch page content with given start and type parameters."""
        params = {
            'start': start,
            'type': type_num
        }
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error fetching page: {e}")
            return ""

    def check_yes_no(self, cell) -> str:
        """Check if a cell contains a yes or no indicator based on CSS classes."""
        yes_span = cell.find('span', class_='catalogue__circle -yes')
        no_span = cell.find('span', class_='catalogue__circle -no')
        
        if yes_span:
            return "Yes"
        elif no_span:
            return "No"
        return ""

    def get_test_link(self, cell) -> str:
        """Extract the href link from the test name cell."""
        link = cell.find('a')
        if link and 'href' in link.attrs:
            return link['href']
        return ""

    def get_test_description(self, test_link: str) -> str:
        """Fetch and extract the description from a test's detail page."""
        if not test_link:
            return ""

        # Construct full URL if it's a relative path
        if test_link.startswith('/'):
            test_link = urllib.parse.urljoin("https://www.shl.com", test_link)

        try:
            logging.info(f"Fetching description for: {test_link}")
            response = requests.get(test_link, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize description parts
            description_parts = []
            
            # Try to find main description
            desc_div = soup.find('div', class_='product-description')
            if desc_div:
                description_parts.append(desc_div.get_text(strip=True))
            
            # Try to find additional details
            details_div = soup.find('div', class_='product-details')
            if details_div:
                description_parts.append(details_div.get_text(strip=True))
            
            # Try to find features
            features_div = soup.find('div', class_='product-features')
            if features_div:
                description_parts.append(features_div.get_text(strip=True))
            
            # Try to find benefits
            benefits_div = soup.find('div', class_='product-benefits')
            if benefits_div:
                description_parts.append(benefits_div.get_text(strip=True))
            
            # Try to find meta description as fallback
            if not description_parts:
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc and 'content' in meta_desc.attrs:
                    description_parts.append(meta_desc['content'])

            # Combine all parts with proper spacing
            full_description = " | ".join(filter(None, description_parts))
            
            time.sleep(1)  # Be respectful with requests
            return full_description

        except requests.RequestException as e:
            logging.error(f"Error fetching description from {test_link}: {e}")
            return ""

    def extract_table_data(self, html_content: str) -> List[Dict]:
        """Extract table data from HTML content."""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        all_data = []
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 4:  # Ensure we have all columns
                    test_link = self.get_test_link(cols[0])
                    data = {
                        'Test Name': cols[0].get_text(strip=True),
                        'Test Link': test_link,
                        'Description': self.get_test_description(test_link),
                        'Remote Testing': self.check_yes_no(cols[1]),
                        'Adaptive/IRT': self.check_yes_no(cols[2]),
                        'Test Type': cols[3].get_text(strip=True)
                    }
                    all_data.append(data)
        return all_data

    def scrape_all_tables(self, max_pages: int = 10):
        """Scrape tables from multiple pages."""
        all_data = []
        
        for start in range(0, max_pages * 12, 12):  # Each page has 12 items
            for type_num in range(1, 9):  # Types 1-8
                logging.info(f"Scraping page with start={start}, type={type_num}")
                
                html_content = self.get_page_content(start, type_num)
                if not html_content:
                    continue
                
                page_data = self.extract_table_data(html_content)
                if page_data:
                    all_data.extend(page_data)
                    logging.info(f"Found {len(page_data)} items on this page")
                
                # Add delay to be respectful to the server
                time.sleep(1)
        
        return all_data

    def save_to_csv(self, data: List[Dict], filename: str = 'shl_products.csv'):
        """Save scraped data to CSV file."""
        if not data:
            logging.warning("No data to save")
            return
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Saved {len(data)} records to {filename}")

def main():
    scraper = SHLScraper()
    logging.info("Starting SHL product catalog scraping...")
    
    data = scraper.scrape_all_tables()
    logging.info(f"Total records scraped: {len(data)}")
    
    scraper.save_to_csv(data)
    logging.info("Scraping completed!")

if __name__ == "__main__":
    main() 