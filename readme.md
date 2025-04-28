# VectorEmbeddings

This project is just for me experimenting with vector embeddings, LLM's, and LangChain.

## Contents

### Ikea catalogue search

The file `ikea_catalogue.py` embeds the Ikea catalogue from 1965 (in `data` directory) in a vector store.

You can then ask questions about contents of the catalogue via console.

The question is embedded and the vector store is queried, the result documents together with the original\
question will be fed into an LLM which tries to answer the question based on the provided documents.

As the Ikea catalogue is in Swedish the question will be translated first.

### Sample furniture search

The file `sample_furniture.py` embeds the products from a static json file (in `data` directory) in a vector store.

You can then send a prompt (e.g. "can you recommend any outdoor furniture?") via console.

First the prompt is expanded into multiple ones and the vector store is queried on each prompt.\
The resulting product documents together with the original prompt are then fed into an LLM to get recommendations.

It is build using LangChain and LCEL, the model prompts are pulled from the hub.

## Requirements

### Install required modules from project root

``pip install -r requirements.txt``

### Set environment variables

```.env
GEMINI_API_KEY="<Your-API-Key>"
```

## Useful links for development

- https://medium.com/data-and-beyond/vector-databases-a-beginners-guide-b050cbbe9ca0
- https://www.datacamp.com/blog/the-top-5-vector-databases?dc_referrer=https%3A%2F%2Fwww.google.com%2F
- https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
- https://docs.trychroma.com/docs/collections/add-data
- https://www.datacamp.com/tutorial/how-to-build-llm-applications-with-langchain
- https://python.langchain.com/docs/tutorials/retrievers/#usage
- https://annacsmedeiros.medium.com/langchain-in-action-translating-summarizing-and-analyzing-text-across-languages-31c28ddf15e6
- https://ikeamuseum.com/en/explore/ikea-catalogue/1965-ikea-catalogue/


- https://python.langchain.com/docs/concepts/lcel/
- https://python.langchain.com/docs/how_to/MultiQueryRetriever/
- https://www.pinecone.io/learn/series/langchain/langchain-expression-language/


- https://tavily.com/
- https://python.langchain.com/docs/how_to/migrate_agent/
- https://gautam75.medium.com/multi-modal-rag-a-practical-guide-99b0178c4fbb
