from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, Any, Optional, TypedDict, Annotated
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool, ToolRuntime
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import BaseMessage, add_messages
from langchain.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import tempfile
import os
import requests

import sqlite3

load_dotenv()

# llm and embedding
llm = ChatOpenAI(model = "gpt-4o-mini")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

# pdf retriever store
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def _get_retriever(thread_id: Optional[str]):
    """Fetch retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_file(file_bytes: bytes, thread_id:str, file_name: Optional[str] = None) -> dict:
    """
    Build FAISS retriver for the uploaded pdf and store it for the thread.
    Return a summary dict that can be surfaced in the UI
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion")

    with tempfile.NamedTemporaryFile(delete = False, suffix= '.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100,separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs= {'k': 5})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "file_name": file_name or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "file_name": file_name or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# tools
search_tool = DuckDuckGoSearchRun(region = 'us-en')

@tool
def calculator(first_num:float, second_num:float, operation:str) -> dict:
    """
    Perform basic arithmatic operations on two numbers
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {'error': "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"unsupported operation: {operation}"}
        return {'first_num': first_num,
                "second_num": second_num,
                "operation": operation,
                "result": result}
    except Exception as e:
        return {'error': str(e)}

@tool
def get_stock_price(symbol:str)-> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)

    return r.json()

@tool
def rag_tool(query: str, runtime: ToolRuntime) -> dict:
    """Retrieve relevant information from the uploaded PDF."""

    thread_id = runtime.config["configurable"]["thread_id"]
    retriever = _get_retriever(str(thread_id))

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first",
            "query": query,
        }

    results = retriever.invoke(query)

    return {
        "query": query,
        "context": [doc.page_content for doc in results],
        "metadata": [doc.metadata for doc in results],
        "source_file": _THREAD_METADATA.get(
            str(thread_id), {}
        ).get("file_name"),
    }


# binding tools with llm
tools = [search_tool, calculator, get_stock_price, rag_tool]
llm_with_tool = llm.bind_tools(tools)

# define state
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# define nodes
def chat_node(state: ChatState, config = None):
    """LLM node responsible for answering the query or requesting a tool call"""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get('thread_id')

        system_message = SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Use the `rag_tool` when the user's question requires information from the uploaded PDF. "
                "Use web search for questions requiring current or external information. "
                "Use the stock price tool for stock-price questions and the calculator for numerical calculations. "
                "If the user asks about the uploaded PDF but no PDF is available, ask them to upload a PDF."
            )
        )

        messages = [system_message, *state['messages']]
        response = llm_with_tool.invoke(messages, config=config)

        return {
            "messages": [response]
        }

tool_node = ToolNode(tools)

# checkpointer
conn = sqlite3.connect(database = "chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn = conn)

# graph 
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer = checkpointer)


# helper functions
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def thread_has_document(thread_id:str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})