from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '25 * 8'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text"""
    return str(len(text.split()))

tools = [calculator, word_count]
tool_map = {t.name: t for t in tools}

llm = ChatOllama(model="llama3.1", temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_fn = tool_map[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append({"role": "tool", "content": str(result), "tool_call_id": call["id"]})
    return {"messages": results}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

task = "What is 25 times 8, and how many words are in 'The quick brown fox jumps over the lazy dog'?"

result = app.invoke({"messages": [HumanMessage(content=task)]})

print("FINAL ANSWER:")
print(result["messages"][-1].content)