from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=groq_api_key
)


wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_tool_raw = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())



def direct_answer_func(query):
    try:
        response = llm.invoke(query).content
        return response
    except Exception:
        return query 
    

direct_answer_tool = Tool(
    name="direct_answer",
    func=direct_answer_func,
    description="Use this for common knowledge, including math (e.g., '2 + 2'), famous people (e.g., 'Who is Einstein?'), and basic facts (e.g., 'What is the capital of France?'). Queries the LLM directly."
)


arxiv_tool = Tool(
    name="arxiv",
    func=arxiv_tool_raw.run,
    description="Use for recent or scientific research topics. Especially when the query includes terms like 'latest research', 'recent papers', or advanced topics like AI, quantum computing, LLMs."
)



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


custom_prompt = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "chat_history"],
    template="""
You are an efficient agent that answers queries by selecting ONE tool.

Previous conversation:
{chat_history}

Available Tools:
- 'direct_answer': For common knowledge, math (e.g., '2 + 2'), famous people (e.g., 'Who is Einstein?'), and basic facts (e.g., 'What is the capital of France?').
- 'wikipedia': For detailed, historical, or current event facts (e.g., 'capital before Tokyo', 'history of Python', '2025 Indian election winner').
- 'arxiv': For recent or scientific research topics (e.g., 'latest research on quantum computing', 'recent AI trends', 'cutting-edge deep learning models').

Rules:
- Select and use ONE tool per query.
- Use 'arxiv' only when the query mentions recent scientific research or a specific research area/topic.
- If a tool returns a clear answer, output it directly.
- If the tool fails or gives no result, try ONE fallback tool or return 'Unable to find answer.'
- Never give long explanations unless asked; be concise and direct (e.g., 'Tokyo').
- Consider previous chat context for follow-up questions.

Scratchpad: {agent_scratchpad}
Query: {input}
"""
)


agent = initialize_agent(
    tools=[direct_answer_tool, wikipedia_tool, arxiv_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  
    max_iterations=5, 
    max_execution_time=120, 
    handle_parsing_errors=True, 
    agent_kwargs={
        "prompt": custom_prompt, 
    },
    memory=memory
)


def run_agent_query(query):
    try:
        response = agent.invoke({"input": query})["output"]
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    

if __name__ == "__main__":
    queries = [
        "Who won the recent presidential election in Indonesia?",# Uses Wikipedia for historical/current events
        "Who is the president of France?",  # Uses Wikipedia for current political information
        "How old is he?", # Follow-up question based on previous context (uses memory)
        "What was the capital of Japan before Tokyo?",  # wikipedia (history)
        "What is the latest research on quantum computing?" ,  # Scientific query, routed to Arxiv
        "What is 5 multiplied by 12?",# Basic math, handled by the direct_answer tool
    ]
    for query in queries:
        print(f"Query: {query}")
        print(f"Response: {run_agent_query(query)}\n")