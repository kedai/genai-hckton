from bs4 import BeautifulSoup

class HTMLContentParser:
    def __init__(self, html_content):
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def get_contents(self):
        """
        Extracts all the main content from the HTML such as headings and paragraphs.
        """
        contents = []
        for tag in self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            content_text = tag.get_text(strip=True)
            if content_text:
                contents.append(content_text)
        return contents

    def get_tables(self):
        """
        Extracts all tables from the HTML and returns them as a list of dictionaries.
        """
        tables = []
        for table in self.soup.find_all('table'):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    rows.append(row_data)
            tables.append({'headers': headers, 'rows': rows})
        return tables

    def get_images(self):
        """
        Extracts all images from the HTML and returns their 'src' and 'alt' attributes.
        """
        images = []
        for img in self.soup.find_all('img'):
            src = img.get('src')
            alt = img.get('alt', '')
            images.append({'src': src, 'alt': alt})
        return images

# Example usage with the provided HTML files
if __name__ == "__main__":
    # Load the HTML content from files
    with open("html/docs_duitnowqr_resources.html", "r", encoding="utf-8") as file:
        html_content_1 = file.read()

    with open("html/docs_credit-transfer_introduction.html", "r", encoding="utf-8") as file:
        html_content_2 = file.read()

    # Create an instance of HTMLContentParser
    parser1 = HTMLContentParser(html_content_1)
    parser2 = HTMLContentParser(html_content_2)

    # Extract contents, tables, and images
    contents_1 = parser1.get_contents()
    tables_1 = parser1.get_tables()
    images_1 = parser1.get_images()

    contents_2 = parser2.get_contents()
    tables_2 = parser2.get_tables()
    images_2 = parser2.get_images()

    # Print results for verification
    print("Contents from File 1:", contents_1)
    print("Tables from File 1:", tables_1)
    print("Images from File 1:", images_1)

    print("Contents from File 2:", contents_2)
    print("Tables from File 2:", tables_2)
    print("Images from File 2:", images_2)
