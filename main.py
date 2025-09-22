from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import wiki_tool, search, save_tool

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(
    model="ai/gemma3:latest",
    base_url="http://127.0.0.1:12434/engines/v1",
    api_key="ignored"
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. You must save the output to a file when you are finished. 
            You must always provide the output in this format and do not provide any additional text: \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search, wiki_tool, save_tool]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research?")

raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
except Exception as e:
    print("Error parsing response", e, "Raw response - ", raw_response)