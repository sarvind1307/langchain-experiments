from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from langchain_openai import ChatOpenAI

import random
from IPython.display import Image, display
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

class State(TypedDict):
  messages: Annotated[list, add_messages]

@tool
def get_stock_price(stock_name: str) -> float:
  """
  Get the stock price of a given stock name.
  Args:
    stock_name: The name of the stock to get the price of.
  Returns:
    The price of the stock.
  """
  return random.uniform(0, 100)

@tool
def buy_stock(stock_name: str, quantity: int, total_cost: float) -> str:
  """
  Buy a given quantity of a given stock.
  Args:
    stock_name: The name of the stock to buy.
    quantity: The quantity of the stock to buy.
    total_cost: The total cost of the stock.
  """
  decision = interrupt(f"Approve the purchase of {quantity} of {stock_name} for a total of {total_cost}?")
  if decision == "yes":
    return f"Bought {quantity} of {stock_name} for a total of {total_cost}"
  else:
    return f"Did not buy {quantity} of {stock_name} for a total of {total_cost}"

tools = [get_stock_price, buy_stock]

llm = init_chat_model("openai:gpt-4.1-nano")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
  return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

image = Image(graph.get_graph().draw_mermaid_png())
with open("graph.png", "wb") as f:
    f.write(image.data)

config = {"configurable": {"thread_id": 1}}

state = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the price of AAPL?"}
        ]
    },
    config=config,
)
print(state["messages"][-1].content)

state = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "Buy 10 stocks of AAPL at current price. What is the total cost?"}
        ]
    },
    config=config,
)
print(state.get("__interrupt__"))
# print(state["messages"][-1].content)
decision = input("Approve the purchase? (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)