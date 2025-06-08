# NOTE: this seems to be mostly hallucinations, GPT was wilding on this task

"""Graph‑RAG: retrieval‑only pipeline backed by Neo4j

Usage
-----
1.  Install deps (tested with llama‑index 0.11.x)::

        pip install llama-index neo4j openai

2.  Export your OpenAI key and Neo4j creds::

        export OPENAI_API_KEY="sk‑…"
        export NEO4J_URI="bolt://localhost:7687"
        export NEO4J_USER="neo4j"
        export NEO4J_PASS="password"

3.  **Build / update** the graph from a directory of text files::

        python graph_rag_retriever.py --build --corpus ./corpus

    This extracts triples, writes them to Neo4j, and saves local index
    metadata under ``./kg_index/`` for fast reloads.

4.  **Query** the graph (no LLM generation)::

        python graph_rag_retriever.py --query "Explain the role of embeddings in RAG"

    The script returns a JSON blob with the top chunks, similarity scores,
    and the Neo4j path that connected them to the query.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import dotenv

from neo4j import GraphDatabase  # type: ignore
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.indices.knowledge_graph.retrievers import (
    KnowledgeGraphRAGRetriever,
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# ---------- Config ---------- #

dotenv.load_dotenv()

# Neo4j credentials fall back to env vars
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Directory where the local part of the index is persisted
PERSIST_DIR = Path("./kg_index")

# Embedding + construction‑time LLM models
EMBED_MODEL = "text-embedding-3-large"
CONSTR_LLM = OpenAI(model="gpt-4o-mini")  # used only for triplet extraction

# Global chunking parameters so we don’t repeat ourselves
Settings.chunk_size = 512
Settings.chunk_overlap = 64


# ---------- Helpers ---------- #

def build_and_save_index(corpus_dir: str) -> None:
    """Ingest documents → build KG → persist to disk + Neo4j."""

    from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
    docs = SimpleDirectoryReader(corpus_dir).load_data()
    docs = SimpleDirectoryReader(corpus_dir).load_data()
    splitter = SentenceSplitter()  # uses global Settings for sizes
    nodes = splitter.get_nodes_from_documents(docs)

    Settings.llm = CONSTR_LLM
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASS,
        url=NEO4J_URI
    )

    kg_index = KnowledgeGraphIndex.from_documents(  # type: ignore[arg-type]
        nodes,
        graph_store=graph_store,
        max_triplets_per_chunk=10,
    )

    # Also save embedding + metadata locally for quick reloads
    kg_index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"✅ Built KG with {len(nodes)} chunks; metadata written to {PERSIST_DIR}")


def init_retriever() -> KnowledgeGraphRAGRetriever:
    """Reload persisted index and return a retriever instance."""
    storage_ctx = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    kg_index = load_index_from_storage(storage_ctx, index_id="knowledge_graph")

    return KnowledgeGraphRAGRetriever(
        index=kg_index,
        graph_store_query_depth=2,
        retriever_mode="hybrid",
        similarity_top_k=8,
    )


def search(query: str) -> List[Dict[str, Any]]:
    retriever = init_retriever()
    results = retriever.retrieve(query)
    return [
        {
            "text": r.node.text,
            "score": r.score,
            "graph_path": r.metadata.get("path"),
            "source_doc": r.node.metadata.get("doc_id"),
        }
        for r in results
    ]


# ---------- CLI ---------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Graph‑RAG retriever CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Ingest corpus and build KG")
    group.add_argument("--query", type=str, help="Run a retrieval query")
    parser.add_argument("--corpus", type=str, default="corpus", help="Path to corpus directory")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.build:
        build_and_save_index(args.corpus)
    else:
        answers = search(args.query)
        print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
