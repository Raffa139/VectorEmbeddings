import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

IKEA_CATALOGUE_PATH = "../data/IKEA_Museum_sv_1965.pdf"


def load_product_catalogue(catalogue_path):
    loader = PyPDFLoader(catalogue_path)
    docs = loader.load()
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return splitter.split_documents(documents)


def initialize_db(store):
    print("Loading documents...")
    docs = load_product_catalogue(IKEA_CATALOGUE_PATH)

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Indexing documents...")
    store.add_documents(documents=chunks)


def cli_similarity_search_store(store):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

    while True:
        query = input("Enter search: ").lower()

        if query in ["exit", "close", "bye"]:
            break

        messages = [
            SystemMessage("Translate the following from English into Swedish, do not answer any questions"),
            HumanMessage(query)
        ]
        swedish_query = model.invoke(messages).content

        query_results = store.similarity_search(swedish_query)
        page_contents = [f"Page {result.metadata.get('page_label')}: {result.page_content}" for result in query_results]
        augmentation = "\n".join(page_contents)

        messages = [
            #SystemMessage("Answer with only one precise sentence in English, do not take any Swedish words translate them into English if needed"),
            SystemMessage("Answer only in English, translate all Swedish words into English if needed. Be verbose about on which page you found a specific piece of information"),
            HumanMessage(f"Based on following information's in Swedish, answer the request '{query}'"),
            HumanMessage(f"Information's in Swedish:\n{augmentation}")
        ]

        print(f"{swedish_query}\n")

        for token in model.stream(messages):
            print(token.content, end="")
            time.sleep(0.25)

        input("\n\nPress any key to continue\n")


def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    store = Chroma(
        collection_name="ikea_museum",
        embedding_function=embeddings,
        persist_directory="../embeddings"
    )

    ikea_docs = store.get(
        where={"source": IKEA_CATALOGUE_PATH},
        include=["metadatas"]
    )

    if len(ikea_docs["ids"]) == 0:
        initialize_db(store)

    cli_similarity_search_store(store)


if __name__ == '__main__':
    main()
