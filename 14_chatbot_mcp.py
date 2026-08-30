from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

import asyncio

load_dotenv()
import requests

HORIZON_API_KEY = os.getenv("HORIZON_API_KEY")

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C://Users//apaks//AppData//Local//Programs//Python//Python313//Scripts//uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "C://Users//apaks//projects//MCP//airthmatic_mcp_server//src//math_mcp_server//main.py"
        ]
    },
    "expense-tracker": {
        "transport": "streamable_http",
        "url": "https://expense-tracker-mcp-ser.fastmcp.app/mcp",
        "headers" : {
            "Authorization": f"Bearer {HORIZON_API_KEY}"
        }
    }
}

client = MultiServerMCPClient(SERVERS)
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        """LLM node that decides if it needs to use llm to generate answer or use a tool"""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)         # tool node is async by default

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke({'messages': [HumanMessage(content = "Show me all the expenses today")]})

    print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())