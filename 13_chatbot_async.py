from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import asyncio

load_dotenv()
import requests

@tool
def calculator(first_num: float, second_num:float, operation:str) -> dict:
    """
    Perform basic arithmatic operation on two numbers.
    Supported operations are add, sub, mul, div
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

tools = [calculator]

llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def build_graph():
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
    chatbot = build_graph()

    result = await chatbot.ainvoke({'messages': [HumanMessage(content = "Find the modulus of 12345 and 23.")]})

    print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())