# RAG-Refiner

GitHub repository with scripts to implement and test a RAG arquitecture on top of an LLM using One Piece chapter summaries as documents.

RAG-Refiner provides scripts for:

- Scraping the One Piece wiki to extract chapter summaries.
- Embedding the scraped documents into a vector database.
- Running interactive inference on the RAG through a CLI, extracting documents from the database and introducing them in the context of every prompt.
- Running exhaustive tests and save the results in a CSV file. This results will contain the time elapsed for each question answered and the Rouge metric score between the generated answers and the ground truths.
- Creating box plots with the results of said tests to analyze the RAG arquitecture's efficiency and effectiveness.

## Getting Started

A version of [Git](https://www.git-scm.com/downloads) should be installed on the system to clone the repository, and you also need to have a version of [Python](https://www.python.org/downloads/) installed to be able to run the scripts.

Start by cloning this repository using `git` and accessing it:

```bash
git clone https://github.com/asiernoide254/rag-refiner
cd RAG-Refiner
```

Then, try to install any dependencies the project needs to run the scripts using `pip`:

```bash
pip install -r requirements.txt
```

It is recommended to use a Python virtual environmente (or "venv") when installing these packages, for better management. You can create a Python virtual environment using the following command:

```bash
py -m venv path/to/new/venv
```

If you are using a *Unix* system, use the `python` command instead of the `py` command.

If you install the package `llama-cpp-python` included in `requirements.txt`, you will only be able to make inference using your CPU. If you want to make inference using your graphics card you can install a version of `llama-cpp-python` with `BLAS` compatibility enabled (refer to this installation guide: <https://python.langchain.com/v0.1/docs/integrations/llms/llamacpp/#installation-with-openblas--cublas--clblast>).

After this is done, simply run the script `setup.py` (you can do this by using the command `py setup.py` on Windows or `python setup.py` on Linux or Mac).

This script will:

1. Create all necessary directories.
2. Extract all data and test examples from the *examples* directory. Generate new data using other scripts if the examples are not present.
3. Download the model llama-2-7b-chat quantized to 4 bits from HuggingFace (<https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf>).

When this script has finished executing you can go ahead and run every other script. The scripts included in the proyect will be shown and explained in the [User Manual](#user-manual).

## User Manual

This manual provides detailed instructions on how to use the scripts `setup.py`, `op_get_long_summaries.py`, `op_get_short_summaries.py`, `embed_documents.py`, `inference.py`, `test_OP_RAG.py`, `process_tests.py` and `analytics.py`. These scripts serve the purpose of generating and using a vector database of documents, evaluating different implementations of RAG arquitectures, and creating graphs with the test results.

Note that the examples of use for every script will be run using the `py` command. This command only works in *Windows* distributions. To execute a Python script in a *Unix* system the `python` command should be used instead.

### setup.py

#### Description

The setup.py script prepares the project structure to directly run the scripts `inference.py`, `test_OP_RAG.py`, `process_tests.py` and `analytics.py`, assuming the necessary dependencies have been previously installed using the `requirements.txt` file. Its functionality includes:

1. Creating the "tests", "models", "data" and "plots" directories.
2. Extracting sample data from the "examples" folder into their respective directories for use by the scripts.
3. If the necessary data is not found in the "examples" folder, it will execute the `op_get_long_summaries.py` and `embed_documents.py` scripts to generate the vector database.
4. Installing the LLaMa 2 7b-chat language model quantized to 4 bits in the "models" folder.

### op_get_long_summaries.py

#### Description

The `op_get_long_summaries.py` script extracts long summaries from the One Piece wiki and stores them in a file called `long_output.txt`.

#### Arguments

- `-o` or `--output`: Directory to save the extracted data file. Defaults to "../data/OP_Output".
- `-l` or `--last_chapter`: Last manga chapter to include in the data extraction. Defaults to 1103.

#### Usage

```bash
py op_get_long_summaries.py [-o output_directory] [-l last_chapter]
```

##### Examples

```bash
py op_get_long_summaries.py
py op_get_long_summaries.py -o /path/to/output -l 1050
```

### op_get_short_summaries.py

#### Description

The `op_get_short_summaries.py` script is identical to `op_get_long_summaries.py` but extracts short summaries from the One Piece wiki instead. The output file for this script will be named `short_output.txt`.

It's arguments and usage are exactly the same as the ones for `op_get_long_summaries.py`.

### embed_documents.py

#### Description

The `embed_documents.py` script generates a vector database from extracted documents. This database is later used to perform queries and retrieve relevant information from the documents.

#### Arguments

- `-d` or `--documents`: Path to the document file. Defaults to "../data/long_output.txt".
- `-p` or `--persist`: Directory to store the vector database. Defaults to "../data/VectorDBLong".

#### Usage

```bash
py embed_documents.py [-d path/to/document] [-p path/to/database]
```

##### Examples

```bash
py embed_documents.py
py embed_documents.py -d /path/to/my/document.txt -p /path/to/my/database
```

### inference.py

#### Description

The `inference.py` script enables an interactive command-line interface for users to interact with the implemented RAG (Retriever-Augmented Generation) architecture. It allows searching and retrieving relevant passages from a vector database to use in a language model.

#### Arguments

- `-m` or `--model`: Path to the model file (must be a GGUF file). Optional, defaults to ../models/llama-2-7b-chat/ggml-model-q4_0.gguf.
- `-v` or `--vector_db_path`: Directory containing the vector database from which to retrieve document passages. Defaults to "../data/VectorDBLong".
- `-c` or `--chunks`: Number of passages to retrieve from the vector database to add to the context. Defaults to 5.
- `search_type`: Subparser to indicate the type of search to perform. Options:
  - `mmr` (Maximal Marginal Relevance)
    - `--fetch_k`: Number of passages to retrieve for MMR.
    - `--lambda_mult`: Lambda multiplier for MMR.
  - `similarity` (default)
  - `similarity_score_threshold`
    - `--score_threshold`: Minimum relevance score threshold for similarity_score_threshold. Required if `search_type` is `similarity_score_threshold`.

#### Usage

```bash
py inference.py [-m path/to/model] [-v path/to/vector_db] [-c number_of_chunks] search_type [--fetch_k number_of_passages] [--lambda_mult lambda_multiplier] [--score_threshold score_threshold]
```

##### Examples

```bash
py inference.py -c 5
py inference.py -m /path/to/my/model.gguf -v /path/to/my/vector_db similarity
py inference.py mmr --fetch_k 10 --lambda_mult 0.5
py inference.py -c 5 similarity_score_threshold --score_threshold 0.75
```

### test_OP_RAG.py

#### Description

The `test_OP_RAG.py` script takes a dataset with questions and answers and evaluates the RAG's responses to the questions against the reference answers. It also calculates the time taken for each question. Then, it creates a new dataset with the score and time taken assigned to each question.

#### Arguments

- `-c` or `--chunks`: Number of passages to retrieve from the vector database to add to the context. Defaults to 5.
- `-n` or `--model_name`: Name of the language model to use. Defaults to "llama-2-7b-chat".
- `-r` or `--results_directory`: Directory to save the test results. Defaults to "../tests/".
- `-m` or `--model_path`: Path to the model file (must be a GGUF file). Defaults to "../models/llama-2-7b-chat/ggml-model-q4_0.gguf".
- `-t` or `--test_dataset_path`: Path to the CSV file containing the questions and answers for the tests. Defaults to "../data/OnePieceQuestionDataset.csv".
- `-v` or `--vector_db_path`: Directory containing the vector database from which to retrieve document passages. Defaults to "../data/VectorDBLong".
- `search_type`: Subparser to indicate the type of search to perform. Options:
  - `mmr` (Maximal Marginal Relevance)
    - `--fetch_k`: Number of passages to retrieve for MMR.
    - `--lambda_mult`: Lambda multiplier for MMR.
  - `similarity` (default)
  - `similarity_score_threshold`
    - `--score_threshold`: Minimum relevance score threshold for similarity_score_threshold. Required if `search_type` is `similarity_score_threshold`.

#### Usage

```bash
py test_OP_RAG.py [-c number_of_chunks] [-n model_name] [-r results_directory] [-m path_to_model] [-t path_to_test_dataset] [-v path_to_vector_db] search_type [--fetch_k number_of_passages] [--lambda_mult lambda_multiplier] [--score_threshold score_threshold]
```

##### Examples

```bash
py test_OP_RAG.py similarity
py test_OP_RAG.py -c 5 -n "other-model" -r /path/to/results -m /path/to/model.gguf -t /path/to/dataset.csv -v /path/to/vector_db
py test_OP_RAG.py mmr --fetch_k 5 --lambda_mult 0.5
py test_OP_RAG.py -c 5 similarity_score_threshold --score_threshold 0.75
```

### process_tests.py

#### Description

The `process_tests.py` script calls the `test_OP_RAG.py` script to perform tests using different parameters and subsequently stores them in separate CSV files in the tests folder.

### analytics.py

#### Description

The `analytics.py` script creates box plots using the previously obtained test results and stores them in a folder called plots. Each plot is saved as a PNG file and includes two box plots: one for the score obtained per question according to the dataset used and another for the time taken per question according to the dataset used.
