import atexit
import asyncio
import os
import sys
import json
from typing import Any, List, Optional
from uuid import uuid4
import pprint
import random

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
from pydantic import PrivateAttr
import duckdb
from pydantic import BaseModel
from langchain.agents.middleware import AgentMiddleware, AgentState, SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver


class State(AgentState):
    context: list[Document]

class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(last_message)
        print("##############################")
        #retrieved_docs = schema_store.similarity_search(last_message.text)

        #docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        #augmented_message_content = (
        #    f"{last_message.text}\n\n"
        #    "Use the following context to answer the query:\n"
        #    f"{docs_content}"
        #)
        #return {
        #    "messages": [last_message.model_copy(update={"content": augmented_message_content})],
        #    "context": retrieved_docs,
        #}


class PlannerResult(BaseModel):
    can_answer: bool
    reason: str
    sql: Optional[str]
    columns_used: List[str]

# 1. Read data CSV with pandas
df_data = pd.read_csv("patients.csv")

# 2. Connect to a DuckDB database file (creates it if not exists)
con = duckdb.connect("data.duckdb")  # this is independent of Chroma

# 3. Write the pandas DataFrame to a DuckDB table

con.execute("DROP TABLE IF EXISTS patients")
con.execute("CREATE TABLE IF NOT EXISTS patients AS SELECT * FROM df_data")

# not used as of now as was detrimental the llm was like ok I only work on those 10 lines
df = con.execute("SELECT * FROM patients LIMIT 10").fetchdf()
csv_string = df.to_csv(index=False)  # no index column


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
    persist_directory="./chroma_duckdb"  # optional but recommended
)


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
        self._tokens = []  # reset buffer

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

schema_store.add_texts(
    texts=texts,
    metadatas=metadatas,
    ids=ids,
)

# --- Tool: same retrieval as minimal, but wrapped for the agent -------------
@tool(response_format="content")
def retrieve_context():
    """
    Return schema information for the clinical-trial `patients` table.
    """
    # Broad schema query – ask for overall schema and fetch many vars
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


@tool(response_format="content_and_artifact")
def run_sql(query: str):
    """
    Execute SQL query against data in DuckDB.
    """
    con = duckdb.connect("data.duckdb")
    try:
        df = con.execute(query).fetchdf()
        if df is not None and len(df) > 0:
            if df.shape == (1, 1):
                col = df.columns[0]
                value = df.iloc[0, 0]
                content = f"SUCCESS: Query result: {col} = {value}"
            else:
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

    return content, df


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
  - Use this to inspect the schema of the patients table.

TOOLS YOU MY USE:
- python_calc()
  - USe this to access numpy as np to perform math calculation or transformation

RULES:

1. Before deciding on SQL, you MUST call `retrieve_context` at least once to
   see the relevant part of the schema. 
   Very important Wait until this tool returned data to you

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


EXECUTOR_PROMPT = """
You are a concise answering assistant for clinical-trial data.
Thought: you should always think about what to do, do not use any tool if it is not needed. 
If you are about to call a tool but you could reasonably answer without it,
do NOT call the tool. Just answer directly.

YOU RECEIVE (as input messages):

- The user's original question.
- The SQL query that need to be executed through run_sql tool
- A brief schema description of the columns involved (optional).

TOOLS YOU MY USE:
- python_calc()
  - USe this to access numpy as np to perform math calculation or transformation


YOUR JOB:

- Run run_sql tool {query}
- Answer the user's question in clear natural language.
- Base your answer ONLY on:
  - the SQL result
  - and any provided schema description
- You MUST NOT use outside medical knowledge or assumptions about ALS.
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

4. If the input indicates an error executing SQL, or if the planner marked
   the question as not answerable, you MUST reply:
   "I don't know based on the provided context."

EXAMPLES:

If you see:
- Question: "What’s the average survival time for male ?"

Then you should answer:
  "The average survival time for male patients is n months based on the provided data."

If you see:
- Planner: can_answer = false
- or SQL result indicates an error or missing data

Then you should answer:
  "I don't know based on the provided context."

"""
checkpointer = InMemorySaver()

debug_handler = DebugHandler(enabled=False)

chat_model = ChatOllama(
    model="qwen3:4b-instruct",        
    base_url="http://localhost:11434",
    temperature=0.0,
)

tools = [retrieve_context, run_sql, python_calc]
agent = create_agent(
    chat_model, 
    tools=tools, 
    middleware=[RetrieveDocumentsMiddleware()],
    checkpointer=checkpointer
    )

thread_id = f"planner-{random.random()}"
# --- Use the agent -----------------------------------------------------------
class MyREPL:
    def __init__(self, agent):
        # Keep a persistent environment between commands
        self.environment = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,
        }
        self.session = PromptSession() 
        self.debug = False
        self.agent = agent

    def is_python_code(self, expression: str) -> bool:
        """
        Very simple heuristic: if it compiles as Python, treat it as Python.
        Falls back to 'exec' if 'eval' doesn't work.
        """
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
            # First try as an expression (so the result can be printed),
            # then fall back to a statement block.
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
        else:
            # Non-Python -> forward to the agent (non-streaming)
            if self.debug:
                print("[DEBUG] calling for planner agent.invoke()")
            plan = agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.replace("{csv_string}", csv_string)},
                        {"role": "user", "content": expression},
                    ]
                },
                config={
                    "callbacks": [debug_handler],
                    "configurable": {              
                        "thread_id": thread_id,
                    },
                },
                
            )
            planner_reply = json.loads([m for m in plan["messages"] if m.__class__.__name__ == "AIMessage"][-1].content)
  
            if not planner_reply['can_answer']:
                return "I don't know based on the provided context."

            executor_result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": EXECUTOR_PROMPT.replace("{query}", planner_reply['sql'])
                        },
                        {
                            "role": "user",
                            "content": (
                                f"USER QUESTION:\n{expression}\n\n"
                            )
                        },
                    ], 
                },
                config={
                    "callbacks": [debug_handler],
                    "configurable": {              
                        "thread_id": thread_id,
                    },
                },
            )



            # `result` is usually a dict with "messages" when using create_agent
            messages = executor_result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                text = getattr(last_msg, "content", str(last_msg))
            else:
                text = str(executor_result)
            if self.debug:
                print("[DEBUG] agent.invoke() returned")
            return text
        
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
    # You’d pass your existing agent instance here
    repl = MyREPL(agent)
    repl.run()
