import atexit
import asyncio
import os
import sys
import json
from typing import Any, List, Optional, TypedDict, Annotated
from uuid import uuid4
import pprint
import random
import operator

import ollama
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from chromadb.config import Settings
from prompt_toolkit import PromptSession
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import PrivateAttr, BaseModel, Field
import duckdb
from langchain.agents.middleware import AgentMiddleware, AgentState, SummarizationMiddleware
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END


# --- Pydantic Schemas and State Definitions ----------------------------------

class PlannerResult(BaseModel):
    can_answer: bool
    reason: str
    sql: Optional[str]
    columns_used: List[str]

class SqlInput(BaseModel):
    sql: str = Field(description="The SQL query to execute against the DuckDB patients table.")

class State(AgentState):
    context: list[Document]

class RetrieveLastAnswerMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        # Note: This middleware is fine for the Executor, but for the new
        # LangGraph implementation, we will manage the history ourselves.
        last_message = state["messages"][-1]

        augmented_message_content = (
            "This was your previous answer:\n"
            f"{last_message.content}\n\n" # Changed .text to .content for BaseMessage compatibility
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": "",
        }

class AgentWorkflowState(TypedDict):
    """
    State for the entire Planner -> Executor workflow, used by LangGraph.
    """
    question: str
    messages: Annotated[list[BaseMessage], operator.add]
    planner_result: Optional[PlannerResult]
    final_answer: Optional[str]


# --- Data and Database Setup ------------------------------------------------

# 1. Read data CSV with pandas
df_data = pd.read_csv("patients.csv")

# 2. Connect to a DuckDB database file (creates it if not exists)
con = duckdb.connect("data.duckdb")

# 3. Write the pandas DataFrame to a DuckDB table
con.execute("DROP TABLE IF EXISTS patients")
con.execute("CREATE TABLE IF NOT EXISTS patients AS SELECT * FROM df_data")
# Close connection after setup
con.close()

# Prepare schema data for Chroma
df_defintion = pd.read_csv("index.csv")
df_defintion = df_defintion.rename(columns={
    "Variable Name": "variable_name",
    "Data Type / Range / Notes": "data_type_notes"
})

def row_to_text(row):
    return (
        f"Variable: {row['variable_name']}. "
        f"Description: {row['Description']}. "
        f"Data type / notes: {row['data_type_notes']}."
    )

texts = [row_to_text(row) for _, row in df_defintion.iterrows()]
metadatas = [
    {
        "variable_name": row["variable_name"],
        "description": row["Description"],
        "data_type_notes": row["data_type_notes"],
    }
    for _, row in df_defintion.iterrows()
]
ids = [row["variable_name"] for _, row in df_defintion.iterrows()]

# --- Embeddings & vector store ----------------------------------------------
embeddings = OllamaEmbeddings(
    model="all-minilm:l6-v2",
)

schema_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_duckdb"
)

# Populate vector store (only if necessary)
if not schema_store.get(ids=ids)['ids']:
    schema_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )


# --- Debug Handler ----------------------------------------------------------
class DebugHandler(BaseCallbackHandler):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._tokens = []

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def _is_enabled(self) -> bool:
        return bool(self.enabled)

    def on_llm_start(self, serialized, prompts, **kwargs):
        if not self._is_enabled():
            return
        print("=== LLM START ===")
        pprint.pp(prompts)
        print("=================")
        self._tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        if not self._is_enabled():
            return
        self._tokens.append(token)
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        if not self._is_enabled():
            return

        full_text = "".join(self._tokens)

        print("\n=== LLM END ===")
        print("Full text:")
        print(full_text)

        print("\nRaw response object:")
        pprint.pp(response)
        print("================")
        self._tokens = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        if not self._is_enabled():
            return
        print("\n=== TOOL CALL ===")
        print("Input:", input_str)
        print("=================")

    def on_tool_end(self, output, **kwargs):
        if not self._is_enabled():
            return
        print("=== TOOL RESULT ===")
        pprint.pp(output)
        print("===================")

debug_handler = DebugHandler(enabled=False)


# --- Tools ------------------------------------------------------------------
@tool(response_format="content")
def retrieve_context():
    """
    Return schema information for the clinical-trial `patients` table.
    """
    # Using a broad query to get the full schema for the planner
    retrieved_docs = schema_store.similarity_search(
        "schema of patients table", k=50
    )

    serialized = "\n\n".join(
        f"Variable: {doc.metadata.get('variable_name', 'UNKNOWN')}\n"
        f"Description: {doc.metadata.get('description', '')}\n"
        f"Data type / notes: {doc.metadata.get('data_type_notes', '')}"
        for doc in retrieved_docs
    )

    return serialized

@tool("python_calc", return_direct=False)
def python_calc(expression: str) -> str:
    """Evaluate a numeric expression using numpy as np in a restricted environment."""
    import numpy as np

    safe_globals = {
        "__builtins__": {},
        "np": np,
    }
    try:
        # Eval ONLY expressions, no statements
        res = eval(expression, safe_globals, {})
        return str(res)
    except Exception as e:
        return f"ERROR: {e}"

@tool(args_schema=SqlInput, response_format="content_and_artifact")
def run_sql(sql: str):
    """
    Execute SQL query against data in DuckDB.
    """
    print(sql)
    con = duckdb.connect("data.duckdb")
    try:
        df = con.execute(sql).fetchdf()
        if df is not None and len(df) > 0:
            if df.shape == (1, 1):
                col = df.columns[0]
                value = df.iloc[0, 0]
                content = f"SUCCESS: Query result: {col} = {value}"
            else:
                # Use to_string for consistent, multi-row display
                content = f"SUCCESS: Query returned {len(df)} rows:\n{df.to_string(index=False)}"
        else:
            content = "SUCCESS: Query returned no results."
    except Exception as e:
        content = (
            f"ERROR: SQL execution failed with error: {e}. "
            "You MUST NOT answer the user's question with a numeric result. "
            "Instead, respond exactly with: \"I don't know based on the provided context.\""
        )
        df = None
    finally:
        con.close()

    # The executor agent will receive content, and the artifact (df) is ignored
    return content, df


# --- Prompts ----------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """
You are an analytical SQL planner for a clinical-trial database.

You NEVER talk to the end-user. Your output is consumed by another system.
You NEVER execute SQL. You ONLY design SQL queries and explain whether a
question can be answered from the data.

DATABASE:

- There is a single table: patients
- Column names and descriptions are defined by the schema.
- You can access the schema using the tool `retrieve_context`.
- You MUST treat the schema as the ONLY source of truth about columns and
  allowed categorical values.


TOOLS YOU MUST USE:

- retrieve_context()
  - This tool returns the COMPLETE schema of the `patients` table.
  - It always returns ALL columns with their descriptions and notes.

TOOLS YOU MY USE:
- python_calc()
  - USe this to access numpy as np to perform math calculation or transformation

RULES:

1. Before deciding on SQL, you MUST call `retrieve_context` exactly once to
   obtain the FULL schema of the `patients` table.
   Do NOT assume any column exists until you have read the tool output.

2. You MUST only use column names that appear EXACTLY in the schema output.
   If a column does not appear there, it DOES NOT exist.

3. For categorical fields, you MUST use the allowed values exactly as given
   in the schema (including capitalization and spelling).

4. You MUST design SQL that uses only the `patients` table.

5. Do NOT try to execute SQL. Do NOT guess numeric results. Do NOT add
   medical or domain knowledge. You are ONLY planning.

YOUR TASK:

Given a single user question in natural language:

1. Think step-by-step (internally) about:
   - What the user is asking.
   - Which columns are needed -> invoke the tool here
   - Which filters / WHERE clauses are needed.
   - Whether the question is actually answerable from the schema.

2. If the question is answerable with a SELECT query, construct a SINGLE
   SQL query that would answer it.

3. If it is not answerable (e.g., required field not in schema, encoding
   ambiguous, etc.), mark it as not answerable.

FINAL OUTPUT FORMAT (VERY IMPORTANT):

You MUST output a single JSON object, and NOTHING else, with the keys:

- can_answer: boolean
- reason: short string explaining your decision
- sql: string or null
- columns_used: array of strings with the column names you used

Examples:


User question:
  "What’s the average survival time for male ?"

Possible output:
{
  "can_answer": true,
  "reason": "reason.",
  "sql": "query",
  "columns_used": ["example_column_1", "example_column_2"]
}

If the question cannot be answered from the schema, use:
{
  "can_answer": false,
  "reason": "No column in the schema encodes the requested concept.",
  "sql": null,
  "columns_used": []
}

Do NOT include explanations outside of this JSON object.
Do NOT talk to the end-user.
Do NOT fabricate numeric answers.
"""

EXECUTOR_PROMPT_TEMPLATE = """
You are a concise answering assistant for clinical-trial data.
Thought: you should always think about what to do, do not use any tool if it is not needed. 
If you are about to call a tool but you could reasonably answer without it,
do NOT call the tool. Just answer directly.

YOU RECEIVE (as input messages):

- The user's original question.
- The SQL query that need to be executed through run_sql tool: {query}
- A brief schema description of the columns involved (optional).

TOOLS YOU MY USE:
- run_sql()
  - Execute the provided SQL query. You MUST use this tool.
- python_calc()
  - USe this to access numpy as np to perform math calculation or transformation

YOUR JOB:

- Run run_sql tool with the provided query.
- Answer the user's question in clear natural language.
- Base your answer ONLY on:
  - the SQL result
  - and any provided schema description
- You MUST NOT use outside medical knowledge or assumptions about the trial.
- If the SQL result does not contain enough information to answer the
  question, you MUST reply exactly with:
    "I don't know based on the provided context."

RULES:

1. NEVER invent or guess numbers. If a numeric value is needed, use ONLY
   the number(s) present in the SQL result text.

2. Do NOT show or explain the SQL unless the user explicitly asks for it.
   Focus on a user-friendly summary.

3. Keep answers short and to the point unless the question clearly asks
   for a detailed explanation.

4. If the SQL result does not contain enough information to answer the
  question, you MUST reply exactly with:
    "I don't know based on the provided context."
"""

# --- Agent Initialization ----------------------------------------------------

checkpointer = InMemorySaver()
executor_planner_id = f"planner-{random.random()}" # Thread ID for Planner's history

chat_model = ChatOllama(
    model="qwen3:4b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)

tools = [retrieve_context, run_sql, python_calc]

# The Planner agent is configured to output the Pydantic model
agent_planner = create_agent(
    chat_model,
    tools=tools,
    checkpointer=checkpointer,
    response_format=ToolStrategy(PlannerResult),
)

# The Executor agent is configured with the history retrieval middleware
agent_executer = create_agent(
    chat_model,
    tools=tools,
    middleware=[RetrieveLastAnswerMiddleware()]
)

# --- LangGraph Node Functions ------------------------------------------------

def plan_query(state: AgentWorkflowState) -> dict:
    """Invokes the Planner agent and updates the state with the PlannerResult."""
    user_question = state["question"]
    if debug_handler.enabled:
        print("[DEBUG] === NODE: plan_query ===")

    # Prepare messages for the planner
    planner_messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    # Invoke the Planner agent
    plan_output = agent_planner.invoke(
        {"messages": planner_messages},
        config={
            "callbacks": [debug_handler],
            "configurable": {"thread_id": executor_planner_id},
        }
    )

    # Extract the structured JSON output
    # NOTE: The last message from a structured output agent is the JSON string
    planner_reply_msg = plan_output["messages"][-1].content
    planner_reply_dict = json.loads(planner_reply_msg)

    # Create the Pydantic object and update state
    planner_result = PlannerResult(**planner_reply_dict)

    # Return the updated state
    return {
        "messages": planner_messages,
        "planner_result": planner_result,
    }

def execute_and_answer(state: AgentWorkflowState) -> dict:
    """Invokes the Executor agent with the planned SQL query."""
    if debug_handler.enabled:
        print("[DEBUG] === NODE: execute_and_answer ===")

    sql_query = state["planner_result"].sql
    user_question = state["question"]

    # 1. Prepare messages for the executor
    executor_prompt = EXECUTOR_PROMPT_TEMPLATE.replace("{query}", sql_query)
    executor_messages = [
        SystemMessage(content=executor_prompt),
        HumanMessage(content=f"USER QUESTION:\n{user_question}\n\n"),
    ]

    # 2. Invoke the Executor agent
    executor_result = agent_executer.invoke(
        {"messages": executor_messages},
        config={"callbacks": [debug_handler]},
    )

    # 3. Extract final answer
    final_text = executor_result.get("messages", [])[-1].content

    # Update state with the final answer
    return {
        "final_answer": final_text,
        "messages": executor_messages # Add executor's messages to history
    }

def fallback_answer(state: AgentWorkflowState) -> dict:
    """Returns a standardized 'cannot answer' message."""
    if debug_handler.enabled:
        print("[DEBUG] === NODE: fallback_answer ===")
    
    # Use the error message required by the prompt
    fallback_msg = "I don't know based on the provided context."
    return {"final_answer": fallback_msg}

# --- LangGraph Routing Function ----------------------------------------------

def route_to_execution(state: AgentWorkflowState) -> str:
    """Conditional routing based on the Planner's 'can_answer' flag."""
    if debug_handler.enabled:
        print("[DEBUG] === ROUTING: route_to_execution ===")

    if state["planner_result"] and state["planner_result"].can_answer:
        # If SQL is generated, proceed to execution
        return "execute"
    else:
        # If not answerable, bypass execution and finish via fallback
        return "fallback"

# --- LangGraph Compilation ---------------------------------------------------

# 1. Initialize the StateGraph 
workflow = StateGraph(AgentWorkflowState)

# 2. Add the nodes
workflow.add_node("plan", plan_query)
workflow.add_node("execute", execute_and_answer)
workflow.add_node("fallback", fallback_answer)

# 3. Set the entry point (always start with planning)
workflow.set_entry_point("plan")

# 4. Define conditional routing from the planner node
workflow.add_conditional_edges(
    "plan",                # Source node
    route_to_execution,    # Router function
    {
        "execute": "execute", # Route to 'execute' if can_answer is True
        "fallback": "fallback", # Route to 'fallback' if can_answer is False
    }
)

# 5. Define the finish points (both execution and fallback lead to the end)
workflow.add_edge("execute", END)
workflow.add_edge("fallback", END)

# 6. Compile the graph
app = workflow.compile(checkpointer=checkpointer)

# --- REPL Class (Simplified) -------------------------------------------------

class MyREPL:
    def __init__(self, compiled_graph):
        self.app = compiled_graph
        self.environment = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,
        }
        self.session = PromptSession()
        self.debug = False

    def is_python_code(self, expression: str) -> bool:
        """Checks if input is valid Python code."""
        try:
            compile(expression, "<repl>", "eval")
            return True
        except SyntaxError:
            try:
                compile(expression, "<repl>", "exec")
                return True
            except SyntaxError:
                return False

    def evaluate(self, expression: str):
        if self.is_python_code(expression):
            # Existing Python REPL logic
            try:
                code = compile(expression, "<repl>", "eval")
            except SyntaxError:
                code = compile(expression, "<repl>", "exec")
                exec(code, self.environment)
                return None
            else:
                return eval(code, self.environment)
        elif expression.startswith("Set Debug"):
            debug = expression.replace('Set Debug ', '', 1).strip()
            self.debug = debug == 'True'
            debug_handler.set_enabled(debug == 'True')
            return f"Debug set to {self.debug}"
        else:
            # Non-Python -> forward to the compiled graph
            if self.debug:
                print("[DEBUG] calling compiled graph.invoke()")

            # Initialize state with the user's question
            initial_state = {
                "question": expression,
                "messages": [HumanMessage(content=expression)],
                "planner_result": None,
                "final_answer": None,
            }

            # Invoke the entire workflow
            result = self.app.invoke(
                initial_state,
                config={
                    "callbacks": [debug_handler],
                    "configurable": {
                        "thread_id": executor_planner_id, # Use planner thread ID for history
                    },
                },
            )

            # The result contains the final state, extract the final answer
            final_text = result["final_answer"]
            if self.debug:
                print("[DEBUG] graph.invoke() returned")

            return final_text

    def run(self):
        print("My Custom REPL - type 'exit' or 'quit' (or Ctrl+D) to quit")

        while True:
            try:
                user_input = self.session.prompt(">> ")

                stripped = user_input.strip()
                if not stripped:
                    continue

                if stripped.lower() in {"exit", "quit", "q"}:
                    break

                result = self.evaluate(user_input)
                if result is not None:
                    print(result)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    repl = MyREPL(app) # Pass the compiled graph
    repl.run()