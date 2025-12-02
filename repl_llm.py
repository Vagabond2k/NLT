import atexit
import asyncio
import os
import sys
from typing import Any, List, Optional
from uuid import uuid4

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

# 1. Read data CSV with pandas
df_data = pd.read_csv("patients.csv")

# 2. Connect to a DuckDB database file (creates it if not exists)
con = duckdb.connect("data.duckdb")  # this is independent of Chroma

# 3. Write the pandas DataFrame to a DuckDB table

con.execute("DROP TABLE IF EXISTS patients")
con.execute("CREATE TABLE IF NOT EXISTS patients AS SELECT * FROM df_data")

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
    def __init__(self, enabled=False):
        self.enabled = enabled

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    # Called when LLM is invoked
    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.enabled:
            print("=== LLM START ===")
            print(prompts)
            print("=================")

    # Called on every streamed token (if streaming)
    def on_llm_new_token(self, token, **kwargs):
        if self.enabled:
            print(f"[token] {token}", end="")

    # Called when model returns
    def on_llm_end(self, response, **kwargs):
        if self.enabled:
            print("\n=== LLM END ===")
            print(response)
            print("================")

    # Called when agent calls a tool
    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.enabled:
            print("=== TOOL CALL ===")
            print("Input:", input_str)
            print("=================")

    def on_tool_end(self, output, **kwargs):
        if self.enabled:
            print("=== TOOL RESULT ===")
            print(output)
            print("===================")

schema_store.add_texts(
    texts=texts,
    metadatas=metadatas,
    ids=ids,
)

# --- Tool: same retrieval as minimal, but wrapped for the agent -------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: Optional[str] = None):
    """
    Return the most relevant schema information for the clinical-trial data.

    If `query` is not provided, default to retrieving the schema
    for the `patients` table.
    """
    # Default query if the model doesn't pass anything
    if not query:
        query = "schema of patients table"

    retrieved_docs = schema_store.similarity_search(query, k=2)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # `serialized` is what the model will see, `retrieved_docs` is kept as artifact
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def run_sql(query: str):
    """
    Execute SQL query against Data in DuckDB
    """
    con = duckdb.connect("data.duckdb")
    try:
        df = con.execute(query).fetchdf()
        if df is not None and len(df) > 0:
            if df.shape == (1, 1):
                col = df.columns[0]
                value = df.iloc[0, 0]
                content = f"Query result: {col} = {value}"
            else:
                content = f"Query returned {len(df)} rows:\n{df.to_string(index=False)}"
        else:
            content = "Query returned no results."
    except Exception as e:
        content = f"SQL execution failed: {e}"
        df = None
    finally:
        con.close()
    return content, df



SYSTEM_PROMPT = """
You are a question-answering assistant for clinical-trial data.

You have access to two tools:

1. retrieve_context: returns the FULL AND ONLY schema of the database.
2. run_sql: executes SQL queries using EXACTLY the fields from the schema.

The database table name is ALWAYS patients.

INTERACTION PROTOCOL (CRITICAL):

There are exactly THREE stages. Decide which stage you are in by looking at the conversation:

STAGE 1 — ONLY retrieve_context:
- Condition: There is NO tool output from retrieve_context in the conversation yet.
- Your response in this stage MUST be:
  - a SINGLE tool call to retrieve_context
  - with NO natural-language text.

STAGE 2 — ONLY run_sql:
- Condition: There IS tool output from retrieve_context,
            but there is NO tool output from run_sql yet.
- In this stage you MUST:
  - read the schema from retrieve_context
  - construct an appropriate SQL query using ONLY the schema fields
  - respond with a SINGLE tool call to run_sql including that SQL
  - and NO other text (no explanation, no comments).

STAGE 3 — FINAL ANSWER (NO MORE TOOLS):
- Condition: There IS tool output from run_sql in the conversation.
- In this stage you MUST:
  - NOT call any more tools
  - NOT output SQL queries as your answer
  - read and interpret the outputs of retrieve_context and run_sql
  - produce a clear natural-language answer for the user.

TOOL / SCHEMA RULES (MUST FOLLOW, NO EXCEPTIONS):

1. Use the schema from retrieve_context as the ONLY source of field names.
   - If a field does not appear in the schema EXACTLY as written, then it DOES NOT EXIST.

2. When translating the user request into SQL, you MUST:
   - Use ONLY the exact field names returned by retrieve_context.
   - NEVER invent, rename, pluralize, simplify, or infer field names.
   - Example: If the schema contains Survival_Time_mo, you MUST NOT replace it with survival_time, survival, or any other variation.

3. You MUST use the patients table in all queries.

4. Your final answer MUST be based ONLY on:
   - the schema from retrieve_context
   - the result of run_sql.

5. If the user's question refers to a concept that does NOT map directly and EXACTLY to fields in the schema, you MUST answer:
   "I don't know based on the provided context."

6. ABSOLUTELY FORBIDDEN:
   - inventing column names
   - guessing which field “probably corresponds”
   - using external medical knowledge
   - answering the question without successful SQL execution
   - performing any mutating action

OUTPUT STYLE FOR FINAL ANSWERS (STAGE 3):

- NEVER return a bare SQL query as the final answer.
- Do NOT show the SQL query unless the user explicitly asks for it.
- When run_sql returns a table with a single numeric value, you MUST:
  - copy that value EXACTLY
  - express it clearly in words.
  Example:
    If run_sql returns a single row with avg(Survival_Time_mo) = 35.9,
    you should answer: "The average survival time is 35.9 months based on the provided data."
- Do NOT convert units. Use the units given in the schema or data.
- If the result is empty or not sufficient to answer, say:
  "I don't know based on the provided context."
"""

debug_handler = DebugHandler(enabled=False)

chat_model = ChatOllama(
    model="llama3.1",           # or "llama3.1:latest"
    base_url="http://localhost:11434",
    temperature=0.0,
)

tools = [retrieve_context, run_sql]
agent = create_agent(chat_model, tools=tools)


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
                print("[DEBUG] calling agent.invoke()")
            result = agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": expression},
                    ]
                },
                config={"callbacks": [debug_handler]},
            )
            # `result` is usually a dict with "messages" when using create_agent
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                text = getattr(last_msg, "content", str(last_msg))
            else:
                text = str(result)
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
