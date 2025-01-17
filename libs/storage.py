from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
import os
from langchain_core.documents import Document


def get_embeddings():
    # create the open-source embedding function
    return OllamaEmbeddings(base_url=os.getenv("OLLAMA_ADDRESS"), model=os.getenv("OLLAMA_EMBEDDING_MODEL"))


def get_vector_store():
    # Connect to local LanceDB
    db_path = os.getenv("DATABASE_DIRECTORY")  # Adjust the path as necessary for your setup
    db = lancedb.connect(uri=db_path)

    # create the open-source embedding function
    embeddings = get_embeddings()

    # Initialize LanceDB vector store
    vs = LanceDB(
        uri=db_path,
        embedding=embeddings,
        table_name=os.getenv("DATABASE_TABLE")
    )

    return vs


def get_doc(content: str):
    doc = Document(page_content=content, metadata={"source": "http://zendit.io"})
    return doc


def load_docs(dataset_folder):
    docs = []  # initialize an empty list to store the documents
    if dataset_folder:  # ensure the dataset folder is defined
        for filename in os.listdir(dataset_folder):  # iterate through each file in the dataset folder
            file_path = os.path.join(dataset_folder, filename)
            if os.path.isfile(file_path):  # ensure the path is indeed a file
                with open(file_path, 'r') as file:
                    content = file.read()
                    final_doc = get_doc(content)  # convert the content into a document
                    docs.append(final_doc)  # append the document to the docs list
    return docs


def cleanup_vector_store():
    # Connect to local LanceDB
    db_path = os.getenv("DATABASE_DIRECTORY")  # Adjust the path as necessary for your setup
    db = lancedb.connect(uri=db_path)
    try:
        db.drop_table(os.getenv("DATABASE_TABLE"))
    except Exception as e:
        print(e)
        pass