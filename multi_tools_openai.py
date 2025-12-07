import duckdb
import pandas as pd
import random
import os
from typing import List, Optional

from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------------------------
# DEBUG HANDLER
# ---------------------------------------------------------------------------

class DebugHandler(BaseCallbackHandler):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.enabled:
            print("\n====== LLM START ======")
            print(prompts)
            print("=======================\n")

    def on_llm_end(self, response, **kwargs):
        if self.enabled:
            print("\n====== LLM END ======")
            print(response)
            print("=====================\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.enabled:
            print("\n====== TOOL CALL ======")
            tool_name = serialized.get("name")
            print("Tool started:", tool_name, "with input", input_str)
            print("=======================\n")

    def on_tool_end(self, output, **kwargs):
        if self.enabled:
            print("\n====== TOOL RESULT ======")
            print(output)
            print("=========================\n")


debug_handler = DebugHandler(enabled=False)


# ---------------------------------------------------------------------------
# 1. Load data + Chroma setup
# ---------------------------------------------------------------------------

df_data = pd.read_csv("patients.csv")

con = duckdb.connect("data43.duckdb")
con.register("df_data", df_data)

con.execute("DROP TABLE IF EXISTS patients")
con.execute("CREATE TABLE patients AS SELECT * FROM df_data")

df_definition = pd.read_csv("index.csv")
df_definition = df_definition.rename(
    columns={
        "Variable Name": "variable_name",
        "Data Type / Range / Notes": "data_type_notes",
    }
)


def row_to_text(row: pd.Series) -> str:
    return (
        f"Variable: {row['variable_name']}. "
        f"Description: {row['Description']}. "
        f"Data type / notes: {row['data_type_notes']}."
    )


texts = [row_to_text(row) for _, row in df_definition.iterrows()]
metadatas = [
    {
        "variable_name": row["variable_name"],
        "description": row["Description"],
        "data_type_notes": row["data_type_notes"],
    }
    for _, row in df_definition.iterrows()
]
ids = [row["variable_name"] for _, row in df_definition.iterrows()]

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

schema_store = Chroma(
    collection_name="patients_schema2",
    embedding_function=embeddings,
    persist_directory="./chroma_schema",
)

schema_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------

@tool("schema_lookup", return_direct=False)
def schema_lookup(columns: Optional[List[str]] = None) -> str:
    """Look up documentation for specific columns in the data dictionary.
    If no columns are provided, return all entries."""
    if not columns:
        return "\n".join(
            f"{row['variable_name']}: {row['Description']} (notes: {row['data_type_notes']})"
            for _, row in df_definition.iterrows()
        )

    subset = df_definition[df_definition["variable_name"].isin(columns)]
    if subset.empty:
        return f"No documentation found for: {columns}"

    return "\n".join(
        f"{row['variable_name']}: {row['Description']} (notes: {row['data_type_notes']})"
        for _, row in subset.iterrows()
    )


@tool("schema_qa", return_direct=False)
def schema_qa(question: str) -> str:
    """Ask a free-text question about the schema / variables using the Chroma
    data dictionary index."""
    docs = schema_store.similarity_search(question, k=5)
    if not docs:
        return "No relevant schema information found."

    parts = []
    for doc in docs:
        md = doc.metadata
        parts.append(
            f"Variable: {md.get('variable_name')}\n"
            f"Description: {md.get('description')}\n"
            f"Notes: {md.get('data_type_notes')}"
        )
    return "\n\n".join(parts)


@tool("peek_rows", return_direct=False)
def peek_rows(limit: int = 10, columns: Optional[List[str]] = None) -> str:
    """Return a small sample of rows so the agent can discover encodings for
    categorical columns such as Gender, Onset_Site, etc."""
    cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols} FROM patients LIMIT {int(limit)}"

    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        return f"ERROR: {e}"

    return df.to_string(index=False) + f"\n\n[showing {min(limit, len(df))} rows]"


@tool("distinct_values", return_direct=False)
def distinct_values(column: str, limit: int = 20) -> str:
    """Return up to `limit` distinct values for a given column.
    Useful to see how categorical variables like Gender are encoded."""
    query = f"SELECT DISTINCT {column} AS value FROM patients LIMIT {int(limit)}"
    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        return f"ERROR: {e}"

    if df.empty:
        return f"No values found for column {column}."

    return df.to_string(index=False)



@tool("duckdb_query", return_direct=False)
def duckdb_query(
    sql: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """Execute a SELECT-only SQL query against the patients DuckDB database.

    You may pass EITHER:
    - sql: the SQL query string, OR
    - query: the SQL query string.

    Only one of these is required.
    """
    statement = sql or query
    if not statement:
        return "ERROR: No SQL query provided. Use either the 'sql' or 'query' argument."

    statement = statement.strip()
    if not statement.lower().startswith("select"):
        return "ERROR: Only SELECT queries are allowed."

    try:
        df = con.execute(statement).fetchdf()
    except Exception as e:
        return f"ERROR: {e}"

    if df.empty:
        return "Query succeeded but returned no rows."

    return df.to_string(index=False)

@tool("column_profile", return_direct=False)
def column_profile(column: str, where: Optional[str] = None) -> str:
    """Compute summary statistics (count, mean, min, max, stddev) for a numeric
    column, optionally filtered with a SQL WHERE clause."""
    query = f"""
        SELECT 
            COUNT(*) AS count,
            AVG({column}) AS mean,
            MIN({column}) AS min,
            MAX({column}) AS max,
            STDDEV_POP({column}) AS stddev
        FROM patients
    """
    if where:
        query += " WHERE " + where

    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        return f"ERROR: {e}"

    return df.to_string(index=False)


@tool("python_calc", return_direct=False)
def python_calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression in a restricted Python
    environment. Use this only for numeric post-processing of tool results."""
    safe_globals = {"__builtins__": {}}
    try:
        res = eval(expression, safe_globals, {})
        return str(res)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# 3. Improved SYSTEM PROMPT (stronger about encodings)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a clinical-trial data assistant.

You have access to the following tools:
- schema_lookup
- schema_qa
- peek_rows
- distinct_values
- duckdb_query
- column_profile
- python_calc

Your purpose is to answer user questions by inspecting the dataset through these tools.
You may reason internally to decide which tools to call, but do NOT reveal chain-of-thought.
Only provide clear conclusions supported by actual tool outputs.

GENERAL PRINCIPLES:

1. Always attempt the necessary tool calls.
   If a question may require data inspection, you must use the tools.

2. Never assume or invent column names.
   Verify column names using schema_lookup or schema_qa before using them
   in SQL queries, column_profile, or any other tool.

3. Never assume categorical encodings.
   Before applying a filter on a categorical column (e.g., Gender, Onset_Site):
   - Call distinct_values(column=...)
     OR
   - Call peek_rows(columns=[...], limit=...)
   Use EXACTLY the encodings returned by these tools.

4. If a query returns empty, NaN-like, or unexpected results:
   - Re-check column names using schema_lookup or schema_qa
   - Re-check categorical encodings with distinct_values or peek_rows
   - Correct assumptions and retry
   Never guess or hallucinate corrections.

5. Never invent numeric values.
   All numeric results must come directly from:
   - duckdb_query
   - column_profile
   - python_calc

6. If the answer cannot be determined after proper tool use,
   reply exactly with:
   "I don't know based on the provided context."

REASONING GUIDELINES:

- Plan tool usage internally; do not expose chain-of-thought.
- Combine multiple tool results when necessary.
- Provide concise, data-grounded answers.
"""


# ---------------------------------------------------------------------------
# 4. LLM + create_agent
# ---------------------------------------------------------------------------

checkpointer = InMemorySaver()

llm = ChatOpenAI(
    model="gpt-5-nano",   # or gpt-4.1, gpt-4o-mini, etc.
    temperature=0.1,
)

tools = [
    schema_lookup,
    schema_qa,
    peek_rows,
    distinct_values,
    duckdb_query,
    column_profile,
    python_calc,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

thread_id = f"planner-{random.random()}"
# ---------------------------------------------------------------------------
# 5. CLI with debug toggle
# ---------------------------------------------------------------------------

def main():
    print("Clinical Trial Agent with Debug + distinct_values.")
    print("Type 'exit' to quit, or:")
    print("  Set Debug True")
    print("  Set Debug False\n")

    while True:
        try:
            q = input(">> ").strip()

            if q.lower() in ("exit", "quit"):
                break

            if q.startswith("Set Debug"):
                # Simple debug toggle
                val = q.replace("Set Debug", "").strip()
                enabled = val.lower() == "true"
                debug_handler.set_enabled(enabled)
                print(f"[DEBUG MODE] -> {enabled}")
                continue

            result = agent.invoke(
                {"messages": [{"role": "user", "content": q}]},
                config={
                    "callbacks": [debug_handler],
                    "configurable": {              
                        "thread_id": thread_id,
                    },
                },
            )

            messages = result.get("messages", [])
            if not messages:
                print("No response from agent.")
                continue

            print("\n--- Answer ---")
            print(messages[-1].content)
            print("--------------\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

