import os
import subprocess
import sys


# Call script test_OP_RAG.py with different parameters
def call_test_script(search_type, argument_value_pairs):
    command = [sys.executable, "test_OP_RAG.py"]
    for arg_name, arg_value in argument_value_pairs.items():
        if arg_name == "chunks":
            command.extend(["-c", str(arg_value)])
    command.append(search_type)
    for arg_name, arg_value in argument_value_pairs.items():
        if arg_name != "chunks":
            command.extend([f"--{arg_name}", str(arg_value)])
    subprocess.run(command)
    return command

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to script directory
os.chdir(script_dir)


# Call the script for search_type "similarity" with different chunk quantities
print("Starting test with similarity search type.\n")
print("------------------------------------------\n")
zero_chunks_command = [sys.executable, "test_OP_vanilla.py"]
print(f"search_type: similarity\nchunks: 0\n\n")
subprocess.run(zero_chunks_command)
for i in [3, 5, 7, 9]:
    print(f"search_type: similarity\nchunks: {i}\n\n")
    call_test_script("similarity", {"chunks": i})


# Call the script for search_type "mmr" with different fetch_k and lambda_mult values
for i in [0.2, 0.4, 0.6, 0.8]: # lambda_mult
    print(f"Starting test with MMR search type and specific fetch_k values (lambda_mult = {i})\n")
    print("------------------------------------------\n")
    for j in [5, 10, 15, 20, 25]: # fetch_k
        print(f"search_type: mmr\nchunks: 5\nlambda_mult: {i}\nfetch_k: {j}\n\n")
        call_test_script("mmr", {"chunks": 5, "fetch_k": j, "lambda_mult": i})


# Call the script for search_type "similarity_score_threshold" with different score_threshold values
print("Starting test with similarity_score_threshold search type and specific score_threshold values\n")
print("------------------------------------------\n")
for i in [0.01, 0.1, 0.2, 0.3, 0.4]:
    print(f"search_type: similarity_score_threshold\nchunks: 5\nscore_threshold: {i}\n\n")
    call_test_script("similarity_score_threshold", {"chunks": 5, "score_threshold": i})
