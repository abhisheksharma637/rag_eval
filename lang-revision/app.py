from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv('.env_rev')
import os
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

os.environ['LANGCHAIN_PROJECT']='lang_rev'
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING']="true"
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY']=os.getenv('ANTHROPIC_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv('HF_KEY')
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')

check_point=InMemorySaver()

llm=ChatGroq(model='openai/gpt-oss-120b')

@tool
def _web_search(qry:str)->str:
    """
    Use this tool for websearch 
    Args: 
        qry :str

    """
    search=TavilySearch()
    qry_rslt=search.invoke(qry)
    top_rslt=qry_rslt['results'][0]['content']
    return top_rslt

system_prompt="You are a helpful AI assistant . Use _web_search tool to do all web searches"

agent=create_agent(model=llm,system_prompt=system_prompt,tools=[_web_search])