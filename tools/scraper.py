# Import necessary modules
import os
import requests  # For downloading images
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup  # For parsing HTML
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service  # Import Service class
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

# List of user agents to randomize requests
user_agents = [
    # Google Chrome on Windows 10
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/98.0.4758.102 Safari/537.36',

    # Safari on macOS Big Sur
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/605.1.15 (KHTML, like Gecko)'
    ' Version/14.0.3 Safari/605.1.15',

    # Mozilla Firefox on Linux
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',

    # Microsoft Edge on Windows 10
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/98.0.1108.43 Safari/537.36 Edg/98.0.1108.43',

    # Chrome on Android
    'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/98.0.4758.102 Mobile Safari/537.36',
]

# List of referrers to randomize requests
referrers = [
    'https://www.google.com/',
    'https://www.bing.com/',
    'https://www.yahoo.com/',
    'https://duckduckgo.com/',
    'https://www.ask.com/',
]

def get_driver():
    # Set up Selenium options for headless browsing
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--disable-gpu")  # Disable GPU acceleration
    options.add_argument("--no-sandbox")  # Bypass OS security model

    # Randomize user agent to make requests less predictable
    user_agent = random.choice(user_agents)
    options.add_argument(f'user-agent={user_agent}')

    # Set up the ChromeDriver Service
    service = Service(ChromeDriverManager().install())

    # Initialize the Chrome WebDriver with service and options
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_docs():
    driver = get_driver()
    base_url = 'https://developer.paynet.my/home/products'
    #base_url = 'https://docs.developer.paynet.my/docs'

    # Create directory 'yyy' if it doesn't exist
    output_dir = 'yyy'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for saving scraped data.")

    # Directory for images
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory '{images_dir}' for saving images.")

    # Initialize sets for tracking visited URLs and a list for URLs to visit
    visited_urls = set()
    urls_to_visit = [base_url]

    print(f"Starting recursive scraping from base URL: {base_url}")

    # Continue scraping until there are no more URLs to visit
    while urls_to_visit:
        current_url = urls_to_visit.pop(0)  # Get the next URL to visit

        # Skip if we've already visited this URL
        if current_url in visited_urls:
            continue

        print(f"\nVisiting: {current_url}")
        visited_urls.add(current_url)  # Mark the URL as visited

        # Randomize referrer
        referrer = random.choice(referrers)
        # Set the referrer in the headers
        driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': {'Referer': referrer}})

        # Visit the URL
        driver.get(current_url)

        # Wait for the page to load
        load_delay = random.uniform(2, 5)
        time.sleep(load_delay)
        print(f"Waited {load_delay:.2f} seconds for page to load.")

        # Wait for specific elements if needed
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'footer'))  # Example element
            )
        except Exception as e:
            print(f"Error loading page {current_url}: {e}")

        # Get the page source
        content = driver.page_source

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # Download images without modifying the HTML
        images = soup.find_all('img')
        for img in images:
            img_url = img.get('src')
            if not img_url:
                continue

            # Resolve relative URLs
            img_url = urljoin(current_url, img_url)

            # Check if image URL is valid and uses HTTP or HTTPS
            img_parsed = urlparse(img_url)
            if img_parsed.scheme.startswith('http'):
                # Download the image
                img_filename = os.path.basename(img_parsed.path)
                local_img_path = os.path.join(images_dir, img_filename)

                # Download and save the image if not already downloaded
                if not os.path.exists(local_img_path):
                    try:
                        img_response = requests.get(img_url, stream=True, timeout=10)
                        if img_response.status_code == 200:
                            with open(local_img_path, 'wb') as img_file:
                                for chunk in img_response.iter_content(1024):
                                    img_file.write(chunk)
                            print(f"Downloaded image: {img_url}")
                        else:
                            print(f"Failed to download image: {img_url} - Status code: {img_response.status_code}")
                    except Exception as e:
                        print(f"Error downloading image {img_url}: {e}")
                        continue
                else:
                    print(f"Image already downloaded: {img_url}")

        # Save content to a file
        # Generate a valid filename from the URL
        path = urlparse(current_url).path
        page_name = path.strip('/').replace('/', '_') or 'index'
        filename = f'{page_name}.html'
        filepath = os.path.join(output_dir, filename)

        # Write the original HTML content to an HTML file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(soup))
                print(f"Saved page to '{filepath}'")
        except Exception as e:
            print(f"Error saving page {current_url}: {e}")

        # Find all links on the current page
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href')

            # Resolve relative URLs
            href = urljoin(current_url, href)

            # Normalize the URL
            href_parsed = urlparse(href)

            # Check if the link is within the same base domain
            if href.startswith(base_url):
                # Avoid re-adding already visited URLs
                if href not in visited_urls and href not in urls_to_visit:
                    urls_to_visit.append(href)
                    print(f"Found new page to visit: {href}")

        # Respectful scraping: Add a delay before the next request
        delay = random.uniform(2, 5)
        print(f"Waiting {delay:.2f} seconds before next request...")
        time.sleep(delay)

    # Close the WebDriver session
    driver.quit()
    print("\nScraping completed. All pages and images have been saved.")

if __name__ == '__main__':
    scrape_docs()

