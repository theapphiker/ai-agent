from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(
    model="ai/gemma3:latest",
    base_url="http://127.0.0.1:12434/engines/v1",
    api_key="ignored"
)

@tool
def multiply_numbers(a:int,b:int) -> int:
    '''Return the product of multiplying two numbers together.'''
    product = a * b
    return product

agent = create_react_agent(llm, [multiply_numbers])
query = "What is 44239554 x 24530?"
response = agent.invoke({"messages": [("human", query)]})
print(response['messages'][-1].content)

message_history = response['messages']

new_query = "How about 23451 and 1234577?"

messages = agent.invoke({"messages": message_history + [("human", new_query)]})

filtered_messages = [msg for msg in messages["messages"] if isinstance(msg, (HumanMessage, AIMessage))
                     and msg.content.strip()]

print({"user_input": new_query, "agent_output":
       [f"{msg.__class__.__name__}:{msg.content}" for msg in filtered_messages]})
