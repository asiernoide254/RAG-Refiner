import os
import argparse
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# Example of use:
# python inference.py --vector_db_path "path/to/specific/vector_db" mmr --fetch_k 3
# python inference.py -c 4 similarity_score_threshold --score_threshold 0.5
# python inference.py -c 7 -m "path/to/model.gguf"


def initialize_model(model_path, persist_directory, chunks, search_type, fetch_k, lambda_mult, score_threshold):
    # For token-wise streaming so you'll see the answer gets generated token by token when Llama is answering your question
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Initializing the LLM
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        temperature=0.1,
        top_p=0.8,
        n_ctx=2048,
        repeat_penalty=1.2,
        last_n_tokens_size=256,
        callback_manager=callback_manager,
        verbose=True
    )

    # Load embeddings and vector database
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.get()

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        '''Use the following pieces of information to answer the user's question. If you don't know the answer just say it, don't try to make up an answer. Answer as concisely and briefly as possible. Don't be verbose.

        Context:
        {context}

        Question: {question}
        Helpful answer: 
        '''
    )

    # Configure search_kwargs depending on search_type and it's conditional parameters
    search_kwargs = {"k": chunks}

    if search_type == "mmr":
        if fetch_k is not None:
            search_kwargs["fetch_k"] = fetch_k
        if lambda_mult is not None:
            search_kwargs["lambda_mult"] = lambda_mult
    elif search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs),
        chain_type_kwargs={"verbose": True, "prompt": prompt_template}
    )

    return qa_chain

def interactive_inference(qa_chain):
    print("Interactive Q&A Session. Press Ctrl+C to exit.")
    try:
        while True:
            question = input("Enter your question: ")
            result = qa_chain.invoke({"query": question})
            print("Answer:", result["result"])
    except KeyboardInterrupt:
        print("\nExiting interactive session.")

if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to script directory
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Initialize model and perform inference interactively")
    parser.add_argument("-m", "--model", default=os.path.join("..", "models", "llama-2-7b-chat", "ggml-model-q4_0.gguf"), help="Path to the model file")
    parser.add_argument("-v", "--vector_db_path", default=os.path.join("..", "data", "VectorDBLong"), help="Directory where the vector database is located")
    parser.add_argument("-c", "--chunks", type=int, default=5, help="Number of chunks to retrieve from the database")
    
    # Add conditional arguments
    # Create subparsers for mmr and similarity_score_threshold search types
    subparsers = parser.add_subparsers(dest="search_type", help="Type of search function for query/document comparison")

    # Subparser for mmr
    parser_mmr = subparsers.add_parser("mmr", help="MMR search type")
    parser_mmr.add_argument("--fetch_k", type=int, default=None, help="Number of items to fetch for MMR")
    parser_mmr.add_argument("--lambda_mult", type=float, default=None, help="Lambda multiplier for MMR")

    # Subparser for similarity_score_threshold
    parser_similarity_score = subparsers.add_parser("similarity_score_threshold", help="Similarity score threshold search type")
    parser_similarity_score.add_argument("--score_threshold", type=float, default=None, help="Score threshold for similarity score (0-1)")

    # Subparser for similarity (default)
    parser_similarity = subparsers.add_parser("similarity", help="Similarity search type")

    args = parser.parse_args()

    # If no subparser was specified, set search_type to "similarity"
    if not args.search_type:
        args.search_type = "similarity"

    # Validate score_threshold if similarity_score search type was selected
    if args.search_type == "similarity_score_threshold":
        score_threshold = args.score_threshold
        if score_threshold is None:
            parser.error("score_threshold must be entered when using the similarity_score_threshold search type")
        elif (score_threshold < 0 or score_threshold > 1):
            parser.error("score_threshold must be between 0 and 1")
    
    qa_chain = initialize_model(
        model_path=args.model,
        persist_directory=args.vector_db_path,
        chunks=args.chunks,
        search_type=args.search_type,
        fetch_k=getattr(args, "fetch_k", None),
        lambda_mult=getattr(args, "lambda_mult", None),
        score_threshold=getattr(args, "score_threshold", None))
    
    interactive_inference(qa_chain)
