import os
import pandas as pd
import matplotlib.pyplot as plt

# Use a serif font
plt.rcParams.update({
    "font.family": "serif"
})

# This function creates a combined graph with two boxplots: one for the rouge score and one for the time elapsed
def create_combined_graph(y_values_rouge, y_values_time, x_values, title, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Rouge Scores subplot
    axs[0].boxplot(y_values_rouge, patch_artist=True)
    axs[0].set_title("Rouge score per question by result dataframe")
    axs[0].set_xlabel("Result dataframes")
    axs[0].set_ylabel("Rouge score")
    axs[0].set_xticklabels(x_values)

    # Add scatter points
    for i, y in enumerate(y_values_rouge):
        x = [i + 1] * len(y)
        axs[0].scatter(x, y, alpha=0.7, edgecolors='black', facecolors='none')

    # Time Elapsed subplot
    axs[1].boxplot(y_values_time, patch_artist=True)
    axs[1].set_title("Time elapsed per question by result dataframe")
    axs[1].set_xlabel("Result dataframes")
    axs[1].set_ylabel("Time elapsed")
    axs[1].set_xticklabels(x_values)

    # Add scatter points
    for i, y in enumerate(y_values_time):
        x = [i + 1] * len(y)
        axs[1].scatter(x, y, alpha=0.7, edgecolors='black', facecolors='none')

    fig.suptitle(title, fontsize=16)
    plt.savefig(save_dir, format='png', dpi=300)
    plt.close(fig)


# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to script directory
os.chdir(script_dir)

# Initialize tests and plots directories
tests_dir = os.path.join("..", "tests")
plots_dir = os.path.join("..", "plots")


# SIMILARITY TESTS
similarity_tests = []
for i in [0, 3, 5, 7, 9]:
    op_test_dataset = pd.read_csv(os.path.join(tests_dir, f"OPTestResults_llama-2-7b-chat_similarity_c{i}.csv"), delimiter=",", encoding="ISO-8859-1")
    similarity_tests.append(op_test_dataset)

rouge_data = [df['Rouge Scores'].values for df in similarity_tests]
time_data = [df['Time Elapsed'].values for df in similarity_tests]
labels = [f'{i} chunks' for i in [0, 3, 5, 7, 9]]

create_combined_graph(
    y_values_rouge=rouge_data,
    y_values_time=time_data,
    x_values=labels,
    title="Similarity search",
    save_dir=os.path.join(plots_dir, "similarity.png")
)


# MMR TESTS
for i in [0.2, 0.4, 0.6, 0.8]:
    mmr_fetch_k_tests = []
    for j in [5, 10, 15, 20, 25]:
        op_test_dataset = pd.read_csv(os.path.join(tests_dir, f"OPTestResults_llama-2-7b-chat_mmr_c5_fetch_k{j}_lambda_mult{i}.csv"), delimiter=",", encoding="ISO-8859-1")
        mmr_fetch_k_tests.append(op_test_dataset)

    rouge_data = [df['Rouge Scores'].values for df in mmr_fetch_k_tests]
    time_data = [df['Time Elapsed'].values for df in mmr_fetch_k_tests]
    labels = [f'fetch_k = {j}' for j in [5, 10, 15, 20, 25]]

    create_combined_graph(
        y_values_rouge=rouge_data,
        y_values_time=time_data,
        x_values=labels,
        title=f"MMR search by fetch_k (lambda_mult = {i})",
        save_dir=os.path.join(plots_dir, f"mmr_fetch_k_lambda_mult{i}.png")
    )


# SIMILARITY SCORE THRESHOLD TESTS
similarity_threshold_tests = []
for i in [0.01, 0.1, 0.2, 0.3, 0.4]:
    op_test_dataset = pd.read_csv(os.path.join(tests_dir, f"OPTestResults_llama-2-7b-chat_similarity_score_threshold_c5_score_threshold{i}.csv"), delimiter=",", encoding="ISO-8859-1")
    similarity_threshold_tests.append(op_test_dataset)

rouge_data = [df['Rouge Scores'].values for df in similarity_threshold_tests]
time_data = [df['Time Elapsed'].values for df in similarity_threshold_tests]
labels = [f'score = {i}' for i in [0.01, 0.1, 0.2, 0.3, 0.4]]

create_combined_graph(
    y_values_rouge=rouge_data,
    y_values_time=time_data,
    x_values=labels,
    title="Similarity Score Threshold search",
    save_dir=os.path.join(plots_dir, "similarity_score_threshold.png")
)
