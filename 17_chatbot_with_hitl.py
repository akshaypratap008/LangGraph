from langgraph.graph import StateGraph, START, END
import requests
from langchain.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

import os
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# tools
@tool
def get_stock_price(symbol:str)-> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)

    return r.json()

@tool
def buy_stocks(stock_price:float, quantity:int, symbol:str):
    """
    Simulate purchasing a given quantity of stocks of a given company.
    Human in the loop: 
    Before executing the purchase this tool will inturrept and wait for human decision ('yes'/anything else)
    """
    # dummy tool

    # pause the flow and seek human decision
    decision = interrupt(f"Approve buying {quantity} stocks of {symbol}? (Yes/No)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Total of {quantity} of {symbol} stocks purachased at a price of {stock_price}",
            "symbol": symbol,
            "quantity": quantity
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} stocks of {symbol} was declined by human",
            "symbol": symbol,
            "quantity": quantity
        }
    

llm = ChatOpenAI(model = "gpt-4o-mini")
tools = [get_stock_price, buy_stocks]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

#checkpointer
checkpointer = MemorySaver()

# graph
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

# ----------UI-----------
if __name__ == "__main__":
    # use the same thread so that the conversation is persisted in memory
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Goodbye")
            break

        # build initial state
        state = {"messages": [HumanMessage(content = user_input)]}

        # run the graph 
        result = chatbot.invoke(
            state, 
            config= {"configurable": {"thread_id": thread_id}}
        )

        # check the HITL interrupt
        interrupts = result.get('__interrupt__', [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")

            decision = input("Your decision: ").strip().lower()

            # resuming the graph after recieving the decision from human
            result = chatbot.invoke(
                Command(resume = decision),
                config = {"configurable": {"thread_id": thread_id}}
            )

        # get the latest message from the assistant
        messages = result['messages']
        last_message = messages[-1]
        print(f"Bot: {last_message.content}\n")