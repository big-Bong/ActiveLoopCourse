import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_key = os.getenv("GOOGLE_CSE_ID")

llm = OpenAI(model_name="text-davinci-003",temperature=0)
prompt = PromptTemplate(input_variables=["query"],
        template="Write a summary of the following text: {query}")
summarize_chain = LLMChain(llm=llm,prompt=prompt)
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for finding information about recent events"
    ),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="Useful for creating summary"
    )
]

agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
response = agent.run("What is the latest news about India's chandrayaan-3?")
print(response)