import argparse
import os
import requests
from bs4 import BeautifulSoup
import time

def modify_html_to_txt(input_html):
    '''
    Extracts the short summary of a One Piece chapter.

    Args:
        input_html (str): The full HTML document for the chapter's Fandom page.

    Returns:
        short_summary (str): String value that contains every paragraph from the short summary of the chapter (one paragraph per line).
    '''
    soup = BeautifulSoup(input_html, 'html.parser')

    # Find summary text
    short_summary = ''

    short_summary_h2 = soup.find('div', class_='mw-parser-output').find_all('h2')

    for h2_element in short_summary_h2:
        # Check if the matching h2 element is found
        if 'Short Summary' in h2_element.get_text():
            # Find all paragraphs after the matching h2 element
            for sibling in h2_element.find_next_siblings():
                if sibling.name == 'p':
                    # Add the text content of each paragraph to short_summary
                    short_summary += sibling.get_text()
                elif sibling.name == 'h2':
                    # Break loop if another h2 element is found
                    break
            break
    
    return short_summary

def save_file(file_directory, paragraph_list):
    '''
    Saves a TXT file with the contents of 'paragraph_list' in the directory 'file_directory'.

    Args:
        file_directory (str): Relative directory of the file to save (including said file).
        paragraph_list (list): List of string values to add to the TXT file.
    
    Returns:
        None
    '''

    with open(file_directory, 'w', encoding='utf-8') as txt_file:
        for chapter in paragraph_list:
            txt_file.write(f"{chapter}\n")
    print(f"TXT file saved in {file_directory}.\n")


def extract_summaries(output_folder, last_chapter):
    # Create the output folder in case it doesn't exist yet
    os.makedirs(output_folder, exist_ok=True)

    # Store every paragraph from the short summary of every chapter in a list
    paragraph_list = []

    for chapter_num in range(1, last_chapter+1):
        # Rest for 10 seconds every 200 chapters to evade getting rate-limited by the API
        if chapter_num % 200 == 0:
            print("Waiting for 10 seconds...")
            time.sleep(10)
            print("Resuming...")
            
        url = f'https://onepiece.fandom.com/wiki/Chapter_{chapter_num}'
        response = requests.get(url)

        paragraph_list.extend(f'Chapter {chapter_num} {modify_html_to_txt(response.text)}'.splitlines())
        print(f"Chapter {chapter_num} added.\n")

    # Write everything on a single TXT file (one line per paragraph)
    file_name = "short_output.txt"
    file_dir = os.path.join(output_folder, file_name)

    # Check if there is not already a file with the same file name on the directory
    if not os.path.exists(file_dir):
        # Save the file if it doesn't exist yet
        save_file(file_dir, paragraph_list)
    else:
        while os.path.exists(file_dir):
            answer = input(f"A file named \"{file_name}\" already exists in \"{output_folder}\". Do you want to replace it? (Y/n): (Otherwise, the file will be saved with another filename)\n").strip().lower()
            if answer == '' or answer == 'y':
                # Replace the existing file
                save_file(file_dir, paragraph_list)
                break
            elif answer == 'n':
                # Save the file with a different file name. Generate a new name based on the existing files
                count = 2
                while True:
                    new_file_dir = f"{file_dir[:-4]}_{count}.txt"
                    if not os.path.exists(new_file_dir):
                        save_file(new_file_dir, paragraph_list)
                        break
                    count += 1
                break
            else:
                print("Invalid answer. Please, answer \"y\" to replace the file, or \"n\" to save with another filename.\n")


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to script directory
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(description="Get short summaries from the One Piece wiki by Fandom")
    parser.add_argument("-o", "--output", default=os.path.join("..", "data", "OP_Output"), help="Directory in which to save the TXT file with the summaries")
    parser.add_argument("-l", "--last_chapter", type=int, default=1103, help="Chapter up to which summaries will be extracted")

    args = parser.parse_args()

    extract_summaries(
        output_folder=args.output,
        last_chapter=args.last_chapter
    )
