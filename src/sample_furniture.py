import os
import json
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever, LineListOutputParser

load_dotenv()

SAMPLE_FURNITURE_PATH = "../data/sample_furniture.json"


def flatten_dict(dictionary: dict) -> dict:
    result = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattened = flatten_dict(value)
            result = {**result, **flattened}
        elif isinstance(value, list):
            joined = ", ".join(value)
            result[key] = joined
        else:
            result[key] = value

    return result


def load_samples(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as file:
        samples = json.load(file)

        docs = []
        for sample in samples:
            title, description, price, additional_info = sample.values()
            metadata = {
                "title": title,
                "price": price,
                "source": path,
                **flatten_dict(additional_info)
            }
            docs.append(Document(page_content=description, metadata=metadata))

        return docs


def initialize_db(store: Chroma):
    print("Loading documents...")
    docs = load_samples(SAMPLE_FURNITURE_PATH)

    print("Indexing documents...")
    store.add_documents(documents=docs)


def format_result_docs(result_docs: list[Document]):
    formatted_contents = []

    for doc in result_docs:
        metadata = doc.metadata
        title = metadata['title']
        price = metadata['price']
        description = doc.page_content
        additional = "; ".join([
            f"{key}={value}"
            for key, value in metadata.items()
            if key not in ["source", "title", "price"]
        ])
        content = f"- {title}  \nID: {doc.id}\n  Description: {description}\n  Price: {price}\n  Additional: {additional}"
        formatted_contents.append(content)

    return "\n".join(formatted_contents)


def cli_similarity_search(store: Chroma, model: BaseChatModel):
    multi_vector_prompt = hub.pull("multi-vector-retriever")
    recommendation_prompt = hub.pull("product-recommendation-engine")

    while True:
        query = input("Enter search: ").lower()
        # query = "i need a tv mount to hang my tv on a wall"
        # query = "im searching for a sturdy oak dining table"
        # query = "can you recommend any outdoor furniture?"

        if query in ["exit", "close", "bye"]:
            break

        multi_query_chain = (
            multi_vector_prompt.partial(n_versions=3)
            | model
            | LineListOutputParser()
        )

        products_chain = (
            MultiQueryRetriever(
                llm_chain=multi_query_chain, retriever=store.as_retriever(), include_original=True
            )
            | format_result_docs
        )

        recommendation_chain = (
            RunnableParallel({"products": products_chain, "request": RunnablePassthrough()})
            | recommendation_prompt
            | model
            | LineListOutputParser()
        )

        recommended_ids = recommendation_chain.invoke(query)

        if recommended_ids:
            docs = store.get_by_ids(recommended_ids)
            print(docs)

        # TODO: Experiment with Products import
        # TODO: Start setup of real project

        #for token in recommendation_chain.stream(query):
        #   print(token.content, end="")
        #   time.sleep(0.25)

        input("\n\nPress any key to continue\n")


def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    store = Chroma(
        collection_name="sample_furniture",
        embedding_function=embeddings,
        persist_directory="../embeddings"
    )

    docs = store.get(
        where={"source": SAMPLE_FURNITURE_PATH},
        include=["metadatas"]
    )

    if len(docs["ids"]) == 0:
        initialize_db(store)

    cli_similarity_search(store, model)


if __name__ == '__main__':
    main()
