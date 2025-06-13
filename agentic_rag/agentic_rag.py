## Experimenting with using multiple agents to refine RAG 

import getpass
import os
import sys
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
from langchain_chroma import Chroma

llm = init_chat_model("gpt-4.1", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# vector_store = InMemoryVectorStore(embeddings)
vector_store = Chroma(
    collection_name="LegalTech_Test",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def generate(): 
    directories = [dir for dir in os.listdir("./corpus_full") if os.path.isdir(f"./corpus_full/{dir}")]
    for dir in directories:

        for filename in os.listdir(f"./corpus_full/{dir}"):
        
            with open(f"./corpus_full/{dir}/{filename}", 'r') as f:
                docs = [Document(page_content=f.read(), from_folder=dir)]
                all_splits = text_splitter.split_documents(docs)
                _ = vector_store.add_documents(documents=all_splits)

def retrieve(question: str, from_dir: str = ""):
    '''Retrieve using RAG from specified directory'''
    retrieved_docs = vector_store.similarity_search(question)
    return {"context": retrieved_docs}

def agentic_retrieve(query):
    '''Retrieve using RAG from document store'''
    ai_response = simple_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Please generate a RAG friendly query for the following question, and output nothing but the new query: {query}" # Note: Later make these instructions more specific to our database and also ask it for metadata information like which directory to search from
            }
        ]
    })
    # Extract the new query from the AI response, handling both dict and message list formats
    if isinstance(ai_response, dict) and "messages" in ai_response and len(ai_response["messages"]) > 0:
        new_query = ai_response["messages"][-1].content
    else:
        new_query = str(ai_response)
    print(f"\nOld query: {query}\nNew query: {new_query}\n\n")
    context = rag_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Please send the user's question to the rag tool. Repeat this process until you get a at least 5 relevant sources. Be critical in what you consider relevant. \
                    Output the sources' text excerpts as well as the document id where they are found. \
                    This is the user's question: {new_query}"
            }
        ]
    })
    return context


memory = MemorySaver()
main_agent = create_react_agent(
    llm,
    tools=[agentic_retrieve],
    prompt="You are a paralegal working at a prestigious Canadian law firm. Always cite relevant sources with direct excerpts and be concise.",
    checkpointer=memory
).with_config(recursion_limit=30)

rag_agent = create_react_agent(
    llm,
    tools=[retrieve],
    prompt = "You are an expert researcher, in charge of collecting sources."
)

simple_agent = create_react_agent(
    llm,
    tools=[],
    prompt = "You are an expert researcher at a law firm, in charge of the RAG system."
)
def main():

    # config = {"configurable": {"thread_id": 1}}
    # for message_chunk, metadata in main_agent.stream({"messages": [
    #     {"role": "user", "content": "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; Does the document indicate that the Agreement does not grant the Receiving Party any rights to the Confidential Information?"}
    # ]}, config, stream_mode="messages"):
    #     print(message_chunk.content, end="")

    config = {"configurable": {"thread_id": 1}}
    response = main_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; Does the document indicate that the Agreement does not grant the Receiving Party any rights to the Confidential Information?"
            }
        ]
    }, config)
    
    # The response is an AddableValuesDict, so extract the final answer from its "messages" key.
    if isinstance(response, dict) and "messages" in response:
        final_message = response["messages"][-1]
        print(final_message.content)
    else:
        print(response)
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "g":
        generate()
    else:
        main()