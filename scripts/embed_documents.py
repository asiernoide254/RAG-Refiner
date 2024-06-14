import os
import argparse
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def embed_documents(documents_path, persist_directory):
    """
    Embeds documents into a vector database.

    Args:
        documents_path (str): Path to the documents file (must be a TXT file).
        persist_directory (str): Directory to persist the vector database.

    Returns:
        None
    """

    # Check if the documents file exists
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"The specified documents file '{documents_path}' does not exist.")

    # Check if the documents file is empty
    if os.path.getsize(documents_path) == 0:
        raise ValueError(f"The specified documents file '{documents_path}' is empty.")

    # Load the documents
    print(f"Loading data from {documents_path}...")
    docs = []
    with open(documents_path, 'r', encoding='utf-8') as file:
        docs = [Document(page_content=line) for line in file]

    # Load embeddings function
    embeddings = HuggingFaceEmbeddings()

    # Split documents into fragments
    print("Splitting documents into fragments...\n")
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separator=''
    )
    all_splits = text_splitter.split_documents(docs)

    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"The vector database was created successfully in {persist_directory}.")

if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to script directory
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Embed documents into a vector database")
    parser.add_argument("-d", "--documents", default=os.path.join("..", "data", "OP_Output", "long_output.txt"), help="Path to the documents file")
    parser.add_argument("-p", "--persist", default=os.path.join("..", "data", "VectorDBLong"), help="Directory to persist the vector database")
    
    args = parser.parse_args()
    
    embed_documents(args.documents, args.persist)
