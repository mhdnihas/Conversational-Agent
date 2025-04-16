# Import required LangChain and Groq modules
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM with Llama 3.1 8B (cloud-based, ~3GB local RAM)
llm = ChatGroq(
    model="llama3-8b-8192",  # Lightweight, free-tier-friendly
    groq_api_key=groq_api_key
)

# Set up Wikipedia tool for detailed or obscure queries
wikipedia_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api)



# Set up Arxiv tool for scientific research queries
arxiv_api = ArxivAPIWrapper()
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)


# Define a direct answer tool for common knowledge and math
# Queries the LLM directly for well-known facts
def direct_answer_func(query):
    try:
        # Call LLM for common knowledge (e.g., capitals, famous people)
        response = llm.invoke(query).content
        return response
    except Exception:
        return query  # Fallback if LLM fails

direct_answer_tool = Tool(
    name="direct_answer",
    func=direct_answer_func,
    description="Use this for common knowledge, including math (e.g., '2 + 2'), famous people (e.g., 'Who is Einstein?'), and basic facts (e.g., 'What is the capital of France?'). Queries the LLM directly."
)



# Initialize memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom prompt to ensure efficient tool choice
custom_prompt = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "chat_history"],
    template=
    """
    You are an efficient agent that answers queries by selecting ONE tool.
    
    Previous conversation:
    {chat_history}
    
    Tools:
    - 'direct_answer': For common knowledge, math (e.g., '2 + 2'), or basic facts (e.g., 'capital of France', 'Tesla stock price').
    - 'wikipedia': For detailed or historical facts (e.g., 'capital before Tokyo', 'history of Python').
    
    Rules:
    - Select and use ONE tool per query.
    - Do NOT reuse a tool if its response is vague or unhelpful.
    - If a tool returns a clear answer (e.g., 'Tokyo'), output it immediately.
    - If a tool fails (e.g., 'No result'), try ONE other tool or output 'Unable to find answer.'
    - Never output explanations unless asked; keep answers short (e.g., 'Tokyo').
    - Consider previous context in the conversation when answering follow-up questions.
    
    Scratchpad: {agent_scratchpad}
    Query: {input}
   """
   )


# Create the agent with tools and prompt
# - Llama 3.1 for reasoning
# - direct_answer for common facts, wikipedia for details
agent = initialize_agent(
    tools=[direct_answer_tool, wikipedia_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Logs for debugging/report
    max_iterations=5,  # Allow correction but prevent loops
    max_execution_time=120, 
    handle_parsing_errors=True, 
    agent_kwargs={
        "prompt": custom_prompt, 
    },
    memory=memory
)


# Function to run queries with error handling
def run_agent_query(query):
    try:
        response = agent.invoke({"input": query})["output"]
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    

if __name__ == "__main__":
    queries = [
        "Who is the president of France?",  # direct_answer (stock)
        "How old is he?",  # direct_answer (geography)
        "What was the capital of Japan before Tokyo?",  # wikipedia (history)
        "What is the latest research on quantum computing?"  # arxiv (science)
    ]
    for query in queries:
        print(f"Query: {query}")
        print(f"Response: {run_agent_query(query)}\n")