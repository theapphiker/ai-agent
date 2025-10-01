from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

memory = MemorySaver()

api_wrapper = WikipediaAPIWrapper(top_k_results=1)

wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def date_checker(date: str):
    """Provide a list of important historical events for a given date in any format."""
    try:
        answer = llm.invoke(f"List important historical events that occurred on {date}.")
        return answer.content
    except Exception as e:
        return f"Error retrieving events: {str(e)}"
    
@tool
def check_palindrome(text: str) -> str:
    """Check if a word or phrase is a palindrome."""

    cleaned = ''.join(char.lower() for char in text if char.isalnum())

    if cleaned == cleaned[::-1]:
        return f"The phrase or word '{text}' is a palindrome."
    else:
        return f"The phrase or word '{text}' is not a palindrome."
    
tools = [wikipedia_tool, date_checker, check_palindrome]

tool_node = ToolNode(tools)

llm = ChatOpenAI(
    model="ai/gemma3:latest",
    base_url="http://127.0.0.1:12434/engines/v1",
    api_key="ignored"
)

model_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    # Check if the last message includes tool calls
    if last_message.tool_calls:
        return "tools"
    
    # End the conversation if no tool calls are present
    return END

def call_model(state: MessagesState):
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:

        return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
    
    return {"messages": [model_with_tools.invoke(state["messages"])]}

workflow = StateGraph(MessagesState)

workflow.add_node("chatbot", call_model)

workflow.add_node("tools", tool_node)

workflow.add_edge(START, "chatbot")
# set up a condition to direct from chatbot to tool or end node
workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
workflow.add_edge("tools", "chatbot")


app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def multi_tool_output(query):
    inputs = {"messages": [HumanMessage(content=query)]}

    for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="", flush=True)
    print("\n")

def user_agent_multiturn(queries):
    for query in queries:
        print(f"User: {query}")

        print("Agent: " + "".join(msg.content for msg, metadata in app.stream(
            {"messages": [HumanMessage(content=query)]}, config, stream_mode="messages") 
            if msg.content and not isinstance(msg, HumanMessage)) + "\n")

queries = ["What happened on the 12 April 1961?", "What about 10 December 1948?",
           "Is `Madam I'm Adam` a palindrome?", "What about `palladium stadium`?"]

user_agent_multiturn(queries)