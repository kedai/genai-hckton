import os
import urllib.parse
from urllib.parse import urljoin, urlparse
from collections import deque
from PyPDF2 import PdfMerger
from tqdm import tqdm
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

base_url = "https://docs.developer.paynet.my/docs"
output_dir = "output"
image_dir = os.path.join(output_dir, "images")

# Create necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Set up Selenium Chrome options
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Initialize the Selenium webdriver
driver = webdriver.Chrome(options=chrome_options)

# Initialize a queue for the URLs to visit
url_queue = deque([base_url])
visited_urls = set()

def generate_filename_from_url(url):
    parsed_url = urlparse(url)
    path = parsed_url.path.replace('/', '_').strip('_')
    #query_fragment = parsed_url.query # or parsed_url.fragment
    #if query_fragment:
    #    path += f"_{query_fragment}"
    #if query_fragment:
    #    path += f"_{query_fragment}"
    return path or "index"

def download_page(url, filename):
    driver.get(url)
    
    # Wait for the page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Hide the footer element before saving the PDF
    driver.execute_script("document.querySelector('footer').style.display = 'none';")

    # Save the page as a PDF using CDP (DevTools Protocol) in landscape mode
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    pdf_data = driver.execute_cdp_cmd("Page.printToPDF", {
        "printBackground": True,
        "preferCSSPageSize": True,
        "landscape": True  # Set the orientation to landscape
    })
    with open(pdf_path, "wb") as f:
        f.write(base64.b64decode(pdf_data['data']))

    # Revert the footer visibility
    driver.execute_script("document.querySelector('footer').style.display = ''")
    
    # Ensure the file was downloaded
    if os.path.exists(pdf_path):
        return pdf_path
    else:
        raise Exception(f"Failed to download PDF for {url}")

def merge_pdfs(pdf_paths, output_path):
    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    merger.write(output_path)
    merger.close()

processed_urls = 0
total_urls = len(url_queue)
successful_downloads = 0
failed_downloads = 0

main_progress = tqdm(total=total_urls, desc="Scraping Progress")

while url_queue:
    url = url_queue.popleft()
    filename = generate_filename_from_url(url)
    pdf_paths = []

    if url in visited_urls:
        continue
    
    try:
        print("Downloading page...")
        pdf_path = download_page(url, filename)
        if os.path.exists(pdf_path):
            pdf_paths.append(pdf_path)
            successful_downloads += 1
            print(f"Successfully downloaded: {filename}.pdf")
        else:
            failed_downloads += 1
            print(f"Failed to download: {filename}.pdf")
    except Exception as e:
        failed_downloads += 1
        print(f"Error downloading {url}: {str(e)}")

    visited_urls.add(url)

    print("Finding links...")
    try:
        link_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
        )
        new_links = 0
        for link in link_elements:
            new_url = urljoin(url, link.get_attribute("href"))
            if new_url not in visited_urls and new_url.startswith(base_url):
                url_queue.append(new_url)
                new_links += 1
        print(f"Found {new_links} new links to process")
    except Exception as e:
        print(f"Error finding links on {url}: {str(e)}")

    if pdf_paths:
        print("Merging PDFs...")
        final_pdf_path = os.path.join(output_dir, f"{filename}_merged.pdf")
        merge_pdfs(pdf_paths, final_pdf_path)
        print("Cleaning up temporary files...")
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    else:
        print(f"No PDFs were downloaded for {url}")

    processed_urls += 1
    main_progress.update(1)
    print(f"Progress: {processed_urls}/{total_urls} pages processed")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

main_progress.close()

print("\nScraping completed!")
print(f"Total pages processed: {processed_urls}")
print(f"Successful downloads: {successful_downloads}")
print(f"Failed downloads: {failed_downloads}")

driver.quit()


