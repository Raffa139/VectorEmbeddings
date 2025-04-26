import os
from dotenv import load_dotenv
from langchain.retrievers.multi_query import LineListOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from pydantic import BaseModel, Field

from src.line_list_tuple_parser import LineListTupleOutputParser

load_dotenv()


def structured_output(query: str, model: BaseChatModel):
    class ResponseFormatter(BaseModel):
        expanded_queries: list[str] = Field(
            description="A list of semantic variations of the original user query")

    structured_model = model.with_structured_output(ResponseFormatter)

    return structured_model.invoke([
        SystemMessage([
            "Given the following user query, expand it to capture a wider range of semantic "
            "variations",
            "Respond with a list of exactly 5 variations using synonyms and broader terms"
        ]),
        HumanMessage(query)
    ]).expanded_queries


def prompt_template_test(model: BaseChatModel):
    multi_vector_prompt = hub.pull("multi-vector-retriever")

    # chain = multi_vector_prompt.partial(n_versions=3) | model | LineListOutputParser()
    chain = (
            RunnableParallel({"question": RunnablePassthrough()})
            | multi_vector_prompt.partial(n_versions=3)
            | model
            | LineListOutputParser()
    )

    answer = chain.invoke("im searching for a sturdy oak dining table")
    print(answer)


def user_request_enhancer(model: BaseChatModel):
    request_enhance_prompt = hub.pull("user-request-enhancer")

    chain = request_enhance_prompt | model | LineListTupleOutputParser()
    answer = chain.invoke("im searching for a sturdy oak dining table")
    print(answer)


def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")
    )

    user_request_enhancer(model)


if __name__ == '__main__':
    main()
