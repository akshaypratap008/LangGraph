from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import requests
import sqlite3

load_dotenv()

# --- llm ---
llm = ChatOpenAI()

# --- tools ----
# search tool
search_tool = DuckDuckGoSearchRun(region = 'us-en')

# calculator
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

# stock price search tool
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
@tool
def get_stock_price(symbol:str)-> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)

    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# --- state --
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# define nodes
def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

# --- checkpointer ---
conn = sqlite3.connect(database = "chatbot.db", check_same_thread= False)
checkpointer = SqliteSaver(conn=conn)

# --- graph structure ---
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# helper function to retrieve all threades
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)