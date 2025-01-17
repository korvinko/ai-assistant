from typing import List
from langchain_ollama.llms import OllamaLLM
import re

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def rerank_documents(
    documents: List[Document],
    query: str,
    model: OllamaLLM,
    top_n: int = 5
) -> List[Document]:
    """
    Reranks documents using the Ollama LLM for relevance scoring.

    Args:
        documents (List[Document]): A list of documents to rerank.
        query (str): The user query.
        model (OllamaLLM): The Ollama LLM instance.
        top_n (int): Number of top documents to return.

    Returns:
        List[Document]: Top N reranked documents.
    """
    if not documents:
        return []

    scored_docs = []

    # Generate relevance scores using LLM
    for doc in documents:
        prompt = f"""
Relevance Scoring:

Query: {query}
Document: {doc.page_content}
On a scale of 1 to 100, how relevant is this document to the query? Provide only the numeric score.
        """
        try:
            response = model.invoke(prompt)  # Use `invoke` instead of `__call__`
            score = extract_number_from_text(response)
            scored_docs.append((doc, score))
        except Exception as e:
            print(f"Error scoring document: {e}")

    # Sort documents by their scores in descending order
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Return the top N documents
    return [doc for doc, _ in ranked_docs[:top_n]]

def extract_number_from_text(text: str) -> float:
    """
    Extracts the first numeric value from a text string.

    Args:
        text (str): Text containing a numeric value.

    Returns:
        float: The extracted numeric value.
    """
    match = re.search(r'\d+(\.\d+)?', text)  # Match integer or decimal numbers
    if match:
        return float(match.group(0))
    raise ValueError(f"No numeric value found in text: {text}")
