# --- 1. Imports ---
# This line imports all functions and classes from the 'rag_system.py' file.
# This likely includes the 'search' and 'load_embedding_matrix' functions we've seen before.
#
# --- Note on 'import *' ---
# While convenient, using 'import *' is generally discouraged in larger projects because it
# can lead to "namespace pollution," where it's unclear which functions come from which module.
# A more explicit and readable approach would be:
# from rag_system import search, load_embedding_matrix
from rag_system import *

# Import the 'json' library for working with JSON data.
import json

# --- 2. Global Setup: Load Embeddings ---
# This line is executed when the script is first loaded. It calls the function to load the
# pre-computed document embeddings from the .npy file into a global variable named 'matrix'.
# By loading this matrix once at the start, subsequent searches can be performed very quickly
# without the need to re-read the file from disk every time.
matrix = load_embedding_matrix("embeddings.npy")


def read_contract(index):
    """
    Reads the content of a specific contract JSON file based on its index.

    Args:
        index (int): The numerical index of the contract file to read (e.g., 0 for 'contract_0.json').

    Returns:
        str: The content of the file as a compact JSON string.
    """
    # Construct the file path dynamically using an f-string.
    file_path = f"data/contract_{index}.json"

    # Use 'with open(...)' for safe file handling. It ensures the file is automatically
    # closed even if errors occur. 'encoding="utf-8"' is crucial for handling a wide range of characters.
    with open(file_path, encoding="utf-8") as jsonf:
        # This line does two things:
        # 1. `jsonf.read()`: Reads the entire content of the file into a string.
        # 2. `json.loads(...)`: Parses the JSON string into a Python dictionary.
        # 3. `json.dumps(...)`: Serializes the Python dictionary back into a JSON string.
        # The purpose of this load-then-dump sequence is to reformat the JSON, typically
        # to ensure it's a valid, compact, single-line string without pretty-printing/indentation.
        return json.dumps(json.loads(jsonf.read()))


def search_contract(query):
    """
    Searches for contracts matching a query and returns the content of the top 3 results.

    Args:
        query (str): The natural language search query.

    Returns:
        list[str]: A list containing the JSON string content of the top 3 matching contracts.
    """
    # 1. Perform the semantic search.
    # The 'search' function (from rag_system.py) takes the query and the embedding matrix.
    # It returns a sorted list of tuples, like [(index, similarity_score), ...].
    # `[:3]` slices this list to keep only the first three elements, which are the top 3 results.
    out = search(query, matrix)[:3]

    # 2. Retrieve the content for the top results.
    # `map()` is a built-in function that applies a given function to each item of an iterable.
    # The `lambda x: read_contract(x[0])` is an anonymous function that takes a result tuple `x`
    # (e.g., (15, 0.85)) and calls `read_contract` using its first element, the index `x[0]`.
    # Finally, `list()` converts the map object into a standard list.
    #
    # --- Alternative using List Comprehension ---
    # A more common and often more readable way to write this in Python is with a list comprehension:
    # return [read_contract(result[0]) for result in out]
    return list(map(lambda x: read_contract(x[0]), out))