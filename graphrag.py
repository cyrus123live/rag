# NOTE: tried to combine knowledge graph and graphrag, but splitting into nodes gave worse results and doing graphrag on unparsed split text gave regular rag again 

from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
graph = Neo4jGraph(refresh_schema=False)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

llm_transformer = LLMGraphTransformer(llm=llm)

docs = []

text_splitter = CharacterTextSplitter.from_tiktoken_encoder()
for filename in os.listdir("./corpus"):
    file = f"./corpus/{filename}"
    with open(file, 'r') as f:
        # documents = [Document(page_content=f.read())]
        # graph_documents = llm_transformer.convert_to_graph_documents(documents)
        docs.append(Document(page_content=filename, metadata={"filename": filename, "doc_type": "file"}))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(f.read())
        for t in texts:
            docs.append(Document(page_content=t, metadata={"filename": filename, "doc_type": "snippet"}))
        # docs.append(Document(page_content=, metadata={"doc_name": filename, "doc_type": "file"}))
        # for node in graph_documents[0].nodes:
        #     docs.append(Document(page_content=node.id, metadata={"from_doc_name": filename, "doc_type": "node"}))

        # documents = [Document(page_content=f.read())]
        # graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # print(f"Nodes:{graph_documents[0].nodes}")
        # print(f"Relationships:{graph_documents[0].relationships}")
        # graph.add_graph_documents(graph_documents, include_source=True)

print("There are", len(docs), "total Documents")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# create the vector store
vectorstore = InMemoryVectorStore(OpenAIEmbeddings())
vectorstore.add_documents(docs)

from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

# retriever = GraphRetriever(
#     store=vectorstore,
#     # edges=[("from_doc_name", "doc_name")],
#     strategy=Eager(start_k=10, adjacent_k=10, select_k=100, max_depth=1),
# )
retriever = vectorstore.as_retriever()

INITIAL_PROMPT_TEXT = "Confidential"
# INITIAL_PROMPT_TEXT = "What are some recommendations of exciting action movies?"
# INITIAL_PROMPT_TEXT = "What are some classic movies with amazing cinematography?"


# invoke the query
query_results = retriever.invoke(INITIAL_PROMPT_TEXT)

# collect the movie info for each film retrieved
compiled_results = {}
for result in query_results:
    if result.metadata["doc_type"] == "file":
        filename = result.metadata["filename"]
        compiled_results[filename] = {
            "filename": filename,
            "snippets": []
        }

# go through the results a second time, collecting the retreived reviews for
# each of the movies
for result in query_results:
    if result.metadata["doc_type"] == "snippet":
        filename = result.metadata["filename"]
        review_text = result.page_content
        if filename not in compiled_results:
            compiled_results[filename] = {
                "filename": filename,
                "snippets": []
            }
        compiled_results[filename]["snippets"].append(review_text)

print(compiled_results)