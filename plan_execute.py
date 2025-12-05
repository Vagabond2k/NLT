import os
# IMPORTANT: Use the corrected import path for the Plan-and-Execute components
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- 1. Define Custom Tools ---

# Tool 1: A simple local tool to replace GoogleSearchTool
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


# Tool 2: The original tool for the second part of the task (shipping)
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
        # A simpler fallback since the task focuses on Dublin
        return f"The estimated shipping cost for {product_name} to {city} is $30.00."

# List of all tools available to the Planner and Executor
tools = [get_current_weather, calculate_shipping_cost]

# --- 2. Initialize Model ---
# Using a powerful model that's good at multi-step reasoning is recommended for P&E
# gpt-4o-mini is an excellent, cost-effective choice for this
llm = ChatOllama(
    model="qwen3:4b-instruct",           # or "llama3.1:latest"
    base_url="http://localhost:11434",
    temperature=0.0,
)

# --- 3. Load Planner and Executor Components ---

# 3a. The Planner (The Brain)
# This component is responsible for analyzing the human input and creating a detailed plan.
planner = load_chat_planner(llm)

# 3b. The Executor (The Worker)
# This component executes the steps defined by the Planner.
executor = load_agent_executor(llm, tools, verbose=True)

# --- 4. Create the Plan-and-Execute Agent ---
pe_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# --- 5. Run the Agent ---
# The task now requires a sequence of two tool calls: Weather -> Shipping.
task = "First, find the current temperature in Dublin. Then, use the city (Dublin) to calculate the express shipping cost for a laptop to that city."

print(f"--- Running Plan-and-Execute Agent for Task: ---\n{task}\n")

# NOTE: The P&E agent uses the 'input' key for the initial query
response = pe_agent.invoke({"input": task})

print("\n\n--- Final Plan-and-Execute Result ---")
print(response["output"])