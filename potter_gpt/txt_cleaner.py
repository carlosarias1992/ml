import re

# The text file you got from the PDF extraction
input_file_path = 'harrypotter.txt' 
# The name of the final, cleaned file
output_file_path = 'harrypotter_cleaned.txt'

def clean_text_file(input_file, output_file):
    """
    Reads a text file, cleans it by removing chapter headers, newlines, 
    and extra spaces, and saves the result to a new file.
    
    Args:
        input_file (str): The path to the text file to clean.
        output_file (str): The path where the cleaned text will be saved.
    """
    try:
        # Step 1: Read the entire content of the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Successfully read '{input_file}'. Starting cleaning process...")

        # Step 2: Remove chapter headers and their following titles.
        pattern = r'CHAPTER\s+([a-zA-Z-]+|\d+)\s*\n.*'
        cleaned_content = re.sub(pattern, '', content)
        
        # Step 3: Replace all newline characters with a single space.
        # This will merge all the text into a single line.
        cleaned_content = cleaned_content.replace('\n', ' ')
        
        # Step 4: Sanitize repeated spaces.
        # This regex finds any occurrence of two or more whitespace characters
        # and replaces it with a single space.
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)
        
        # Finally, remove any leading/trailing whitespace from the whole text block
        cleaned_content = cleaned_content.strip()
        
        # Step 5: Write the fully cleaned content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        print(f"✅ Cleaning complete! Sanitized text saved to '{output_file}'")

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the script ---
if __name__ == "__main__":
    clean_text_file(input_file_path, output_file_path)
