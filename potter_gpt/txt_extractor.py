import pdfplumber
from tqdm import tqdm # Import the tqdm library

# --- Configuration ---
# Set the path to your PDF file
pdf_path = 'harrypotter.pdf'
# Set the name for your output text file
output_txt_path = 'harrypotter.txt'


def extract_text_with_progress(pdf_file, output_file):
    """
    Extracts text from a PDF while showing a progress bar for each page processed.
    
    Args:
        pdf_file (str): The path to the input PDF file.
        output_file (str): The path to the output .txt file.
    """
    try:
        full_text = ""
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            
            # Wrap pdf.pages with tqdm to create the progress bar
            # desc is the description, and unit is the label for each iteration
            page_iterator = tqdm(pdf.pages, desc=f"Extracting pages from {pdf_file}", unit="page")
            
            # Loop through each page using the new iterator
            for page in page_iterator:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        # Write the extracted text to the output file
        print("\nWriting text to output file...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"✅ Successfully extracted text to '{output_file}'")

    except FileNotFoundError:
        print(f"❌ Error: The file '{pdf_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Run the script ---
if __name__ == "__main__":
    extract_text_with_progress(pdf_path, output_txt_path)
