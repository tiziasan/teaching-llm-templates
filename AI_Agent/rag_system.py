# --- Import Necessary Libraries ---
# NumPy is fundamental for numerical operations, especially for handling the embedding vectors as arrays.
import numpy as np

# Scikit-learn is a powerful machine learning library.
# We use TSNE for dimensionality reduction to visualize high-dimensional embeddings in 2D.
from sklearn.manifold import TSNE
# We use cosine_similarity to measure how similar the query vector is to the document vectors.
from sklearn.metrics.pairwise import cosine_similarity

# Matplotlib is used for plotting and creating the 2D visualization.
import matplotlib.pyplot as plt

# FlagEmbedding provides an easy-to-use interface for state-of-the-art text embedding models.
from FlagEmbedding import BGEM3FlagModel

# os and json are standard Python libraries for interacting with the file system and parsing JSON files, respectively.
import os
import json

# --- 1. Model Initialization ---
# We are using the BGE-M3 model from the Beijing Academy of Artificial Intelligence (BAAI).
# BGE-M3 is a powerful multilingual and multi-functional text embedding model. It excels at
# creating dense vector representations (embeddings) that capture the semantic meaning of text.
# It's a strong choice for tasks like semantic search, clustering, and classification.
# For more details, see: https://huggingface.co/BAAI/bge-m3
#
# `use_fp16=True` enables half-precision floating-point format (float16).
# This significantly reduces GPU memory usage and can speed up computation on compatible hardware
# (like NVIDIA's Tensor Core GPUs) with a minimal, often negligible, loss in precision.
print("Loading BGE-M3 model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
# --- Alternative Models ---
# Other powerful embedding models could be used here, such as:
# - Sentence-Transformers models (e.g., 'all-mpnet-base-v2'): Great for general-purpose English sentence embeddings.
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer('all-mpnet-base-v2')
# - Cohere's models (e.g., 'embed-english-v3.0'): High-performance commercial models.
# - OpenAI's models (e.g., 'text-embedding-3-large'): Another strong commercial option.
print("Model loaded successfully.")


def load_contracts_from_folder(folder_path):
    """
    Loads all '.json' files from a specified folder, extracts subject and body,
    and returns them as a list of formatted strings.

    Args:
        folder_path (str): The path to the directory containing the contract JSON files.

    Returns:
        list: A list of strings, where each string is the concatenated content of a contract.
              Returns an empty list if the directory is not found or is empty.
    """
    # Basic error handling to ensure the provided path is a valid directory.
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return []

    documents = []
    print(f"Reading contract files from '{folder_path}'...")
    # Using sorted() ensures a consistent order of documents, which is crucial because
    # the index of an embedding will correspond to the index in this sorted list.
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # We format the text to provide more context to the embedding model.
                    # Including field names like "Subject:" can help the model better
                    # understand the structure and importance of different parts of the text.
                    full_text = f"Subject: {data.get('subject', '')}\n\n{data.get('body', '')}"
                    documents.append(full_text)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: Could not read file {filename}. Error: {e}")

    print(f"Found and loaded {len(documents)} documents.")
    return documents


def index_database(data, output_path='embeddings.npy'):
    """
    Generates embeddings for a list of documents and saves them to a file.
    This process is often called "indexing" in the context of search systems.

    Args:
        data (list): A list of document strings to be embedded.
        output_path (str): The file path where the generated embeddings will be saved.
    """
    print("Generating embeddings for the database...")
    # The model.encode() method takes a list of strings and returns a dictionary.
    # We are interested in 'dense_vecs', which are the primary embeddings for semantic similarity.
    # This operation can be computationally intensive and is best run on a GPU.
    embeddings = model.encode(data)['dense_vecs']

    # We save the embeddings as a .npy file. This is NumPy's native binary format,
    # which is highly efficient for storing and loading large numerical arrays.
    np.save(output_path, embeddings)
    print(f"Embeddings saved to '{output_path}'")


def load_embedding_matrix(embeddings_path):
    """
    Loads a pre-computed embedding matrix from a .npy file.

    Args:
        embeddings_path (str): The path to the .npy file containing the embeddings.

    Returns:
        np.ndarray: The loaded matrix of embeddings.
    """
    print(f"Loading embeddings from '{embeddings_path}'...")
    loaded_embeddings = np.load(embeddings_path)
    return loaded_embeddings


def search(query, embedding_matrix):
    """
    Performs a semantic search for a given query against a matrix of document embeddings.

    Args:
        query (str): The search query text.
        embedding_matrix (np.ndarray): The pre-computed embeddings of the document database.

    Returns:
        list: A list of tuples, where each tuple contains the document index and its
              cosine similarity score to the query, sorted in descending order of similarity.
    """
    # 1. Encode the user's query into the same vector space as the documents.
    # We access ['dense_vecs'] because the model can output multiple embedding types.
    # We select [0] because we are only encoding a single query.
    query_embedding = model.encode([query])['dense_vecs'][0]

    # 2. Calculate the cosine similarity between the query embedding and all document embeddings.
    # Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 to 1.
    # It effectively measures orientation, not magnitude, making it robust to differences in document length.
    # A score of 1 means semantically identical (in the model's view), 0 means unrelated, and -1 means opposite.
    #
    # --- Alternative Similarity Metrics ---
    # - Dot Product: `np.dot(embedding_matrix, query_embedding)`. Faster to compute but is sensitive to
    #   vector magnitude (longer or more "intense" text might get higher scores, which can be undesirable).
    # - Euclidean Distance: `np.linalg.norm(embedding_matrix - query_embedding, axis=1)`. Measures the straight-line
    #   distance between vectors. It's less common for semantic search as it's not normalized in the same way as cosine similarity.
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]

    # 3. Sort the results to find the most similar documents.
    # We use `enumerate` to pair each similarity score with its original document index.
    # The `lambda` function tells `sorted` to use the similarity score (the second element, x[1]) as the sorting key.
    # `reverse=True` ensures that the highest similarity scores are ranked first.
    similarity_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return similarity_results


def visualize_space_query(data, query, embedding_matrix):
    """
    Visualizes the high-dimensional embedding space in 2D using t-SNE,
    highlighting the position of the query relative to the documents.

    Args:
        data (list): The original list of document strings (used for labeling, not computation).
        query (str): The search query string.
        embedding_matrix (np.ndarray): The matrix of document embeddings.
    """
    print("Generating 2D visualization with t-SNE...")
    # First, create an embedding for the query.
    query_embedding = model.encode([query])['dense_vecs'][0]

    # Combine the document embeddings and the query embedding into a single matrix for t-SNE processing.
    # The query embedding is appended as the last row.
    jointed_matrix = np.vstack([embedding_matrix, query_embedding])

    # --- t-SNE Parameter Tuning ---
    # `perplexity` is a crucial t-SNE parameter related to the number of nearest neighbors each point considers.
    # A common rule of thumb is to set it between 5 and 50. It must be less than the number of samples.
    # We dynamically set it to avoid errors when the dataset is very small.
    num_samples = len(jointed_matrix)
    perplexity_value = min(30, num_samples - 1)

    # Handle the edge case where there are not enough samples for t-SNE.
    if perplexity_value <= 0:
        print("Cannot generate visualization with 1 or fewer data points.")
        return

    # Initialize the t-SNE model.
    # `n_components=2`: We want a 2D plot.
    # `random_state=42`: Ensures reproducibility of the plot. t-SNE has a random component.
    # `init='pca'`: Initializes the embedding with Principal Component Analysis, which is a stable and often faster starting point.
    # `learning_rate='auto'`: A new feature in recent scikit-learn versions that adapts the learning rate, generally leading to better results.
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')

    # Fit and transform the combined matrix to get the 2D coordinates.
    embeddings_2d = tsne.fit_transform(jointed_matrix)

    # --- Plotting ---
    plt.figure(figsize=(12, 10))

    # Plot all the document embeddings. `embeddings_2d[:-1]` selects all rows except the last one.
    plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], c='blue', edgecolor='k', alpha=0.6, label='Contracts')

    # Plot the query embedding, which is the last row. We make it larger and red to stand out.
    plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], c='red', s=150, edgecolor='k', label='Query')

    # Add a text label for the query to make the plot easier to interpret.
    plt.text(embeddings_2d[-1, 0] + 0.1, embeddings_2d[-1, 1] + 0.1, query, fontsize=10, color='red', weight='bold')

    # Add plot titles and labels for clarity.
    plt.title('t-SNE Visualization of Contract Embeddings and Query')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.legend()
    plt.show()


# --- Main Execution Block ---
# This `if` statement ensures that the code inside it only runs when the script is executed directly
# (not when it's imported as a module into another script).
if __name__ == "__main__":

    # Define constants for file paths. This is good practice for configurability.
    DATA_FOLDER = 'data'
    EMBEDDINGS_FILE = 'embeddings.npy'

    # --- Workflow Step 1: Load Data ---
    contract_documents = load_contracts_from_folder(DATA_FOLDER)

    # Proceed only if documents were successfully loaded.
    if contract_documents:
        # --- Workflow Step 2: Index Data ---
        # This step is computationally expensive and should only be run when the underlying data changes.
        # In a real-world application, you might add logic to check if the data has changed
        # before re-generating the embeddings.
        index_database(contract_documents, output_path=EMBEDDINGS_FILE)

        # --- Workflow Step 3: Load Embeddings for Search ---
        embedding_matrix = load_embedding_matrix(EMBEDDINGS_FILE)

        # --- Workflow Step 4: Perform Search ---
        # Define a query to search for.
        my_query = "Find claims related to water damage"
        search_results = search(my_query, embedding_matrix)

        # --- Workflow Step 5: Display Results ---
        # Print the top 5 most relevant documents found.
        print(f"\n--- Top 5 Search Results for query: '{my_query}' ---")
        for i, (doc_index, similarity) in enumerate(search_results[:5]):
            print(f"\nRank {i + 1} (Similarity: {similarity:.4f})")
            print("-" * 20)
            # Retrieve the original document text using the index from the search results.
            print(contract_documents[doc_index])
            print("-" * 20)

        # --- Workflow Step 6: Visualize ---
        # Generate and show the t-SNE plot to visualize the relationships.
        visualize_space_query(contract_documents, my_query, embedding_matrix)
    else:
        print("\nExecution stopped because no documents were found.")