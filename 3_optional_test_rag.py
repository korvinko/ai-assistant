import nest_asyncio
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from libs.storage import get_vector_store, rerank_documents
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel

nest_asyncio.apply()

load_dotenv()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# Predefined query
query = "Tell me about security features. Provide all information."

# Prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know; don't try to make up an answer. Be as concise as possible, but provide all details if the user asks.
{context}
Question: {question}. This question is related to the service zendit.io. Provide the URL to the documentation next to the provided information. Return output in markdown format.
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

vs = get_vector_store()

llm = OllamaLLM(
    model=os.getenv("OLLAMA_MAIN_MODEL"),
    base_url=os.getenv("OLLAMA_ADDRESS"),
    temperature=0.1,
)

async def main():
    try:

        messages = [Message(role="user", content=query)]
        formatted_prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        retriever = vs.as_retriever()
        docs = await retriever.ainvoke(formatted_prompt)
        docs = await rerank_documents(docs)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = QA_CHAIN_PROMPT.format(context=context, question=formatted_prompt)

        async for chunk in llm.astream(final_prompt):
            print(chunk, end="", flush=True)

    except Exception as e:
        print(f"Error: {e}")


loop = asyncio.get_event_loop()
if loop.is_running():
    asyncio.ensure_future(main())
else:
    asyncio.run(main())
