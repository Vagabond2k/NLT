import os
# CORRECT FIX: The installed package 'langchainhub' exposes the module 'langchain_hub'.
import langchainhub as hub
# FIX 2: AgentExecutor is deprecated but lives in langchain_core.agents
from langchain_core.agents import AgentExecutor 
# FIX 3: Use the single available agent constructor in your installation
from langchain.agents.factory import create_agent 

# Standard core imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# --- 1. Define Custom Tools ---

@tool
def get_current_weather(city: str) -> str:
    """
    Returns the current weather conditions (temperature and general status) for a specified city.
    Requires city (e.g., Dublin) as input.
    """
    city = city.lower()
    if "dublin" in city:
        return "The current temperature in Dublin is 10°C and it is overcast."
    elif "paris" in city:
        return "The current temperature in Paris is 15°C and it is sunny."
    else:
        return f"Weather data for {city} is not available in our local system."

@tool
def calculate_shipping_cost(product_name: str, city: str) -> str:
    """
    Calculates the estimated shipping cost for a given product to a destination city.
    Requires product_name (e.g., laptop) and city (e.g., Dublin) as input.
    """
    product_name = product_name.lower()
    city = city.lower()
    
    if "laptop" in product_name and "dublin" in city:
        return "The estimated express shipping cost for a laptop to Dublin is $45.00."
    else:
        return f"The estimated shipping cost for {product_name} to {city} is $30.00."

tools = [get_current_weather, calculate_shipping_cost]

# --- 2. Initialize Model and Prompt ---
llm = ChatOllama(
    model="qwen3:4b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Pull the standard ReAct/Tool-Calling prompt from the LangChain Hub
prompt = hub.pull("hwchase17/openai-tools-agent") 

# --- 3. Create the Agent ---
# The ReAct logic is embedded within the 'create_agent' function using LangGraph.
agent = create_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# --- 4. Create the Agent Executor ---
# AgentExecutor runs the 'agent' object (which is a runnable LangGraph sequence).
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# --- 5. Run the Agent ---
task = "First, find the current temperature in Dublin. Then, use the city (Dublin) to calculate the express shipping cost for a laptop to that city."

print(f"--- Running ReAct Agent for Task: ---\n{task}\n")

# NOTE: The agent now expects a 'messages' key containing a list of BaseMessage objects.
response = agent_executor.invoke({
    "messages": [HumanMessage(content=task)]
})

print("\n\n--- Final ReAct Agent Response ---")
print(response["messages"][-1].content)