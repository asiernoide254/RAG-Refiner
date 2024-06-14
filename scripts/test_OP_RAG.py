import argparse
import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pandas import read_csv
from time import time
from rouge import Rouge
import re

# Example of use:
# python test_OP_RAG.py --vector_db_path "path/to/specific/vector_db" mmr --fetch_k 3
# python test_OP_RAG.py -c 4 similarity_score_threshold --score_threshold 0.5
# python test_OP_RAG.py -c 7 -m "path/to/model.gguf"


# Function to preprocess the text. This is very useful when comparing answers to apply the Rouge metric
def preprocess_text(text):
    """
    Function to preprocess the text. Takes a string value and returns the same string in lowercase without hyphens, puntuation marks or multiple consecutive spaces.
    
    Args:
        text (str): The text to be preprocessed.
        
    Returns:
        str | None: The preprocessed text (or None if the text to process is None).
    """
    if text is None:
        return None
    else:
        # Replace hyphens with spaces
        text = re.sub(r'-', ' ', text)
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text).lower()
        # Remove additional spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

rouge = Rouge()

# Function to calculate rouge scores
def calculate_rouge(model_answer, reference_answer):
    if not model_answer or model_answer == "":  # If hypothesis is empty
        return 0.0
    score_collection = rouge.get_scores(model_answer, reference_answer)
    return score_collection[0]['rouge-l']['f']


def test_model(model_path, vectordb_path, results_path, chunks, search_type, model_name, test_dataset_path, fetch_k, lambda_mult, score_threshold):
    """
    Function to test the model against a test dataset.
    
    Args:
        model_path (str): Path to the GGUF file of the language model.
        vectordb_path (str): Directory where the vector database is located.
        results_path (str): Directory in which to save the test results.
        chunks (int): Number of chunks to retrieve from the vector database.
        search_type (str): Search type.
        model_name (str): Name of the language model.
        test_dataset_path (str): Path to the CSV file containing questions and answers for testing.
        fetch_k (int, optional): Number of items to fetch for MMR. Defaults to None.
        lambda_mult (float, optional): Lambda multiplier for MMR. Defaults to None.
        score_threshold (float, optional): Score threshold for similarity score. Defaults to None.
    
    Returns:
        None
    """
    # Load the LLM
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=32,
        temperature=0.1,
        top_p=0.8,
        n_ctx=2048,
        repeat_penalty=1.2,
        last_n_tokens_size=256
    )

    # Load embeddings function
    embeddings = HuggingFaceEmbeddings()

    # Load vectordb from disk
    vectordb = Chroma(persist_directory=vectordb_path, embedding_function=embeddings)
    vectordb.get()

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

    # Metrics evaluation
    # Load the testing QA dataset
    op_test_dataset = read_csv(test_dataset_path, delimiter=";", encoding="ISO-8859-1")

    # Add LLM answers, time elapsed and rouge scores to the loaded dataset
    answers = []
    time_elapsed = []
    rouge_scores = []

    question_index = 1

    for question, ground_truth in zip(op_test_dataset['Question'], op_test_dataset['Answer']):
        print(f"Question: {question}")
        time_0 = time()
        answer = qa_chain.invoke({"query": question})
        answer_time = time() - time_0
        respuesta = answer["result"]
        rouge_score = calculate_rouge(preprocess_text(respuesta), preprocess_text(ground_truth))

        time_elapsed.append(answer_time)
        answers.append(respuesta)
        rouge_scores.append(rouge_score)

        print(f"Elapsed time: {answer_time}")
        print(f"Answer: {respuesta}")
        print(f"Rouge Score: {rouge_score}")
        print(f"Progress: {question_index/len(op_test_dataset['Question'])*100}%\n\n")
        question_index += 1

    op_test_dataset['Model Answer'] = answers
    op_test_dataset['Time Elapsed'] = time_elapsed
    op_test_dataset['Rouge Scores'] = rouge_scores


    print(op_test_dataset)

    print(f"Average time elapsed per question: {op_test_dataset['Time Elapsed'].mean()} seconds")
    print(f"Average Rouge score: {op_test_dataset['Rouge Scores'].mean()}")
    print(f"Median Rouge score: {op_test_dataset['Rouge Scores'].median()}")
    print(f"Standard deviation of Rouge score: {op_test_dataset['Rouge Scores'].std()}")
    print(f"Total time elapsed: {round(op_test_dataset['Time Elapsed'].sum(), 0)} seconds")

    # Generating name for the CSV file containing the result set
    results_file_name = f"OPTestResults_{model_name}_{search_type}_c{chunks}"
    if search_type == "mmr":
        if fetch_k is not None:
            results_file_name += f"_fetch_k{fetch_k}"
        if lambda_mult is not None:
            results_file_name += f"_lambda_mult{lambda_mult}"
    elif search_type == "similarity_score_threshold":
        results_file_name += f"_score_threshold{score_threshold}"
    results_file_name += ".csv"

    # Check if the file already exists
    results_file_full_path = os.path.join(results_path, results_file_name)
    while os.path.exists(results_file_full_path):
        answer = input(f"A file named \"{results_file_name}\" already exists in \"{results_path}\". Do you want to replace it? (Y/n): (Otherwise, the file will be saved with another filename)\n").strip().lower()
        if answer == '' or answer == 'y':
            # Replace the file
            print("Replacing the file...")
            op_test_dataset.to_csv(results_file_full_path)
            break
        elif answer == 'n':
            # Save the file with a different file name. Generate a new name based on the existing files
            count = 2
            while True:
                new_results_file_name = f"{results_file_name[:-4]}_{count}.csv"
                new_results_file_full_path = os.path.join(results_path, new_results_file_name)
                if not os.path.exists(new_results_file_full_path):
                    print(f"Saving file as \"{new_results_file_name}\"...")
                    op_test_dataset.to_csv(new_results_file_full_path)
                    break
                count += 1
            break
        else:
            print("Invalid answer. Please, answer \"y\" to replace the file, or \"n\" to save with another filename.\n")

    if not os.path.exists(results_file_full_path):
        # Save the file in case it doesn't exist yet
        print(f"Storing results dataset in \"{results_file_full_path}\"...")
        op_test_dataset.to_csv(results_file_full_path)

if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to script directory
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Test the RAG against a test dataset.")
    parser.add_argument("-c", "--chunks", type=int, default=5, help="Number of chunks to retrieve from the vector database")
    parser.add_argument("-n", "--model_name", default="llama-2-7b-chat", help="Name of the language model")
    parser.add_argument("-r", "--results_directory", default=os.path.join("..", "tests"), help="Directory in which to save the test results")
    parser.add_argument("-m", "--model_path", default=os.path.join("..", "models", "llama-2-7b-chat", "ggml-model-q4_0.gguf"), help="Path to the GGUF file of the language model")
    parser.add_argument("-t", "--test_dataset_path", default=os.path.join("..", "data", "OnePieceQuestionDataset.csv"), help="Path to the CSV file containing questions and answers for testing")
    parser.add_argument("-v", "--vector_db_path", default=os.path.join("..", "data", "VectorDBLong"), help="Directory where the vector database is located")

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


    test_model(
        model_path=args.model_path,
        vectordb_path=args.vector_db_path,
        results_path=args.results_directory,
        chunks=args.chunks,
        search_type=args.search_type,
        model_name=args.model_name,
        test_dataset_path=args.test_dataset_path,
        fetch_k=getattr(args, "fetch_k", None),
        lambda_mult=getattr(args, "lambda_mult", None),
        score_threshold=getattr(args, "score_threshold", None)
    )

