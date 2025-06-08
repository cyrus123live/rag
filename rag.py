## FROM https://python.langchain.com/docs/tutorials/rag/

import getpass
import os
from pdfminer.high_level import extract_text
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text = extract_text('sample.pdf')
docs = [Document(page_content=text)]
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

def retrieve(question: str):
    '''Retrieve using RAG from document store'''
    retrieved_docs = vector_store.similarity_search(question)
    return {"context": retrieved_docs}

memory = MemorySaver()
graph = create_react_agent(
        ChatOpenAI(model="o4-mini-2025-04-16"),
        tools=[retrieve],
        prompt=f"",
        checkpointer=memory
    ).with_config(recursion_limit=30)

config = {"configurable": {"thread_id": 1}}

for message_chunk, metadata in graph.stream({"messages": [
    {"role": "user", "content": "Please use the retreieve tool and then tell me about System 2 Attention."}
]}, config, stream_mode="messages"):
    print(message_chunk.content, end="")

print("\n")
