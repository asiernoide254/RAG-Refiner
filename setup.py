import os
import subprocess
import sys
import zipfile
import shutil
from huggingface_hub import hf_hub_download


def create_directory(directory):
    """
    Create a directory if it does not exist.

    Args:
        directory (str): Path of the directory to be created.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def extract_files(extract_directory, zip_file_path):
    """
    Extract the contents of a ZIP file into a directory. Returns None if the ZIP file couldn't be found and the ZIP's file path otherwise.

    Args:
        extract_directory (str): Path of the directory where the ZIP file will be extracted to.
        zip_file_path (str): Path of the ZIP file to be extracted.

    Returns:
        None | str
    """
    # Check if the ZIP file exists
    if os.path.exists(zip_file_path):
        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(extract_directory)
            print(f"File '{zip_file_path}' extracted to '{extract_directory}'.\n")
            return zip_file_path
    else:
        print(f"File '{zip_file_path}' does not exist.\n")
        return None

def run_script(script_path):
    """
    Safely runs a Python script and waits for it to finish the execution.

    Args:
        script_path (str): Path of the python script to run.

    Returns:
        None
    """
    # Ensure the script path is absolute
    script_path = os.path.abspath(script_path)

    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Script {script_path} does not exist.")
        return

    # Run the script
    try:
        process = subprocess.Popen([sys.executable, script_path])
        process.communicate()  # Wait for the script to finish
    except Exception as e:
        print(f"Error running script {script_path}: {e}")

# Get script full directory
script_dir = os.path.dirname(os.path.abspath(__file__))
    
# Change to the script's directory
os.chdir(script_dir)

# Create directories 'tests', 'models', and 'data'
create_directory('tests')
create_directory('models')
create_directory('data')

# Paths for ZIP file and extraction
# Comprobar si existe VectorDBLong
zip_file_path_data = os.path.join('examples', 'data.zip')
extract_path_data = os.path.join('data')
zip_file_path_tests = os.path.join('examples', 'tests.zip')
extract_path_tests = os.path.join('tests')

# Exctracting zip files to their respective folders. Executing neccesary scripts to generate data for long summaries if they're not found.
if extract_files(extract_directory=extract_path_data, zip_file_path=zip_file_path_data) is None:
    print("Running op_get_long_summaries.py to create file long_output.txt on the data folder.\n")
    run_script(os.path.join("scripts", "op_get_long_summaries.py"))
    print("Running embed_documents.py to create vector database (VectorDBLong) on the data folder.\n")
    run_script(os.path.join("scripts", "embed_documents.py"))

extract_files(extract_directory=extract_path_tests, zip_file_path=zip_file_path_tests)

# Extract LLM for inference

# Destination directory
destination_dir = os.path.join("models", "llama-2-7b-chat")
destination_file = os.path.join(destination_dir, "ggml-model-q4_0.gguf")

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Download the model
hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGUF", filename="llama-2-7b-chat.Q4_K_M.gguf", local_dir=destination_dir)

try:
    os.rename(os.path.join(destination_dir, "llama-2-7b-chat.Q4_K_M.gguf"), destination_file)
    print(f"Model file renamed to {destination_file}\n")
except OSError as e:
    print(f"Error al renombrar el archivo: {e.strerror}\n")


# Remove the automatically created huggingface directory after installing the model
if (os.path.exists(os.path.join(destination_dir, ".huggingface"))):
    shutil.rmtree(os.path.join(destination_dir, ".huggingface"))
    print(f"Residual files removed successfully.\n")
