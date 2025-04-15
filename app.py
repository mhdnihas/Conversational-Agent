# Import LangChain and Groq modules
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()


# Access the API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")


# Initialize Groq LLM (Llama 3.1 8B for efficiency)
llm = ChatGroq(
    model="llama3-8b-8192",  
    groq_api_key=groq_api_key  
)

# Set up Wikipedia tool
wikipedia_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api)



# Create the agent
# - Uses Groq’s Llama 3.1 for reasoning
# - Queries Wikipedia for factual data when needed
agent = initialize_agent(
    tools=[wikipedia_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # Shows reasoning steps
)


# Function to run queries
def run_agent_query(query):
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    
# Test the agent
if __name__ == "__main__":
    query = "Tell me about the history of artificial intelligence"
    print(f"Query: {query}")
    print(f"Response: {run_agent_query(query)}")