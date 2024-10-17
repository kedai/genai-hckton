from bs4 import BeautifulSoup
import os
import json

# Directory containing HTML files
directory = 'html'

# Function to extract data from an HTML file
def extract_data_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        data = {
            'header': None,
            'footer': None,
            'navigation': None,
            'content_texts': [],
            'content_tables': [],
            'content_images': []
        }

        # Extract header (navigation bar)
        header = soup.find('nav')
        if header:
            data['header'] = str(header)

        # Extract footer
        footer = soup.find('footer')
        if footer:
            data['footer'] = str(footer)

        # Extract navigation
        navigation = soup.find('nav')
        if navigation:
            data['navigation'] = str(navigation)

        # Extract content texts
        main_content = soup.find('main')
        if main_content:
            for paragraph in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                data['content_texts'].append(paragraph.get_text(strip=True))

        # Extract tables
        tables = main_content.find_all('table') if main_content else []
        for table in tables:
            data['content_tables'].append(str(table))

        # Extract images
        images = main_content.find_all('img') if main_content else []
        for img in images:
            data['content_images'].append(img.get('src'))

        return data

# Loop through all HTML files and extract data
extracted_data = {}
for filename in os.listdir(directory):
    if filename.endswith('.html'):
        file_path = os.path.join(directory, filename)
        extracted_data[filename] = extract_data_from_html(file_path)

# Save extracted data to a JSON file
output_file = 'output/extracted_content.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(extracted_data, json_file, indent=4)

print(f"Extracted data has been saved to {output_file}")
