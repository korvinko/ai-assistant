from fastapi import HTTPException
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from fastapi.responses import HTMLResponse, JSONResponse
import markdown
from libs.storage import get_vector_store, rerank_documents
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import string
from typing import List
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update as necessary for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve HTML and other static files
app.mount("/web", StaticFiles(directory="./static"), name="index.html")

# Define request body
class QueryRequest(BaseModel):
    query: str
    historyKey: str

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know; don't try to make up an answer. Be as concise as possible, but provide all details if the user asks.
{context}
Question: {question}. This question is related to the service zendit.io. Provide the URL to the documentation next to the provided information. Return output in markdown format.
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Set up the vector store and model
vc = get_vector_store()
# Initialize memory for conversation
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MAIN_MODEL"),
    base_url=os.getenv("OLLAMA_ADDRESS"),
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.1,
)

def generate_history_key(length=12):
    characters = string.ascii_letters + string.digits  # You can also add string.punctuation for special characters
    return ''.join(random.choice(characters) for _ in range(length))

@app.post("/ask")
def ask(query_request: QueryRequest):
    try:
        memory = ConversationBufferMemory(memory_key=query_request.historyKey, return_messages=True)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vc.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            memory=memory
        )

        result = qa_chain({"query": query_request.query})
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define a message format for conversation
class Message(BaseModel):
    role: str  # 'system', 'user', or 'assistant'
    content: str

# Define request body
class QueryRequestV2(BaseModel):
    query: List[Message]  # Expect a list of messages
    historyKey: str = ""

@app.post("/completion")
async def completions(request: Request):
    try:
        body_json = await request.json()
        prompt = body_json.get("prompt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str("Error parsing JSON body:" + str(e)))

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vc.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            chain_type="stuff",
        )

        result = qa_chain({"query": prompt})
        resp = result.get("result")

        # Convert Markdown to HTML
        html_resp = markdown.markdown(
            resp,
            extensions=["extra", "sane_lists"]  # Better Markdown list and formatting handling
        )

        # Convert Markdown to HTML
        html_resp = markdown.markdown(resp)

        return HTMLResponse(content=html_resp, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tags")
def ollamaTags():
    try:
        # Define the object to return
        response_data = {
            "models": [
                {
                    "name": "ai-assistant",
                    "model": "ai-assistant",
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": {},
                }
            ]
        }
        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False


@app.post("/v1/chat/completions")
async def ollamaChat(request: ChatRequest):
    try:
        messages = request.messages
        latest_user_message = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)

        if not latest_user_message:
            raise HTTPException(status_code=400, detail="No user messages found in the request.")

        retriever = vc.as_retriever()
        docs = await retriever.ainvoke(latest_user_message)
        docs = await rerank_documents(docs)
        context = "\n\n".join([doc.page_content for doc in docs])

         # Format the final prompt with the latest user message, combined with context
        final_prompt = QA_CHAIN_PROMPT.format(context=context, question=latest_user_message)


        async def event_stream():
            async for chunk in llm.astream(final_prompt):
                response = {
                    "id": "chatcmpl-81",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.getenv("OLLAMA_MAIN_MODEL"),
                    "system_fingerprint": "fp_ollama",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": chunk
                            },
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(response)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
