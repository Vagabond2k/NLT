import pandas as pd
import ollama
import pandasai as pai
from pandasai import Agent
from pandasai.llm.base import LLM
from pandasai.core.prompts.base import BasePrompt
import sys
import atexit
import os
from prompt_toolkit import PromptSession


class OllamaLLM(LLM):
    """
    Simple PandasAI-compatible LLM that talks to a local Ollama server.
    """

    def __init__(self, model: str = "llama3.1:latest", host: str = "http://localhost:11434", **kwargs):
        # LLM base takes api_key + **kwargs, but we don't need an API key for Ollama
        super().__init__(api_key=None, **kwargs)
        self.model = model
        self.client = ollama.Client(host=host)

    @property
    def type(self) -> str:
        # Just an identifier string for logging/metadata
        return "ollama"

    def call(self, instruction: BasePrompt, context=None) -> str:
        """
        This is the one abstract method we *must* implement.

        `instruction` is a BasePrompt and has `.to_string()`.
        `context` (if not None) has `.memory`, which LLM.base knows how to use.
        """
        prompt = instruction.to_string()
        
        guard = """
        YOU ARE A PYTHON CODE GENERATOR FOR PANDASAI USING SQL + DUCKDB.

        RULES (MUST FOLLOW ALL):

        1. DO NOT modify any base tables (no CREATE, DROP, INSERT, UPDATE, DELETE).
        - Only use SELECT queries.

        2. Each question must be answered based on the full table,
        unless the user explicitly asks for a filter (e.g. "only smokers").
        Do NOT permanently restrict future queries.

        3. If you need a subset, use SELECT with WHERE in the SQL query,
        but do NOT assume future questions are restricted to that subset.

        4. ALWAYS return the final answer via a variable named `result`:
        result = {"type": "<type>", "value": <value>}
        Never use print() for the final answer.

        5. Do NOT change global Python variables that affect future questions.
        """


        if context is not None and getattr(context, "memory", None) is not None:
            prompt = self.prepend_system_prompt(prompt, context.memory)
        # print("===== PROMPT SENT TO OLLAMA =====")
        # print(prompt)
        # print("=================================")


        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": guard},
                {"role": "user", "content": prompt},
            ],
        )

        return response["message"]["content"]


# -------------------------
# Load data
# -------------------------
df_data = pd.read_csv("patients.csv")

dict_df = pd.read_csv("index.csv")
field_descriptions = dict_df.set_index("Variable Name")[
    ["Description", "Data Type / Range / Notes"]
].apply(
    lambda x: f"{x['Description']} (Data Type/Range: {x['Data Type / Range / Notes']})",
    axis=1,
).to_dict()

# -------------------------
# PandasAI Agent
# -------------------------
local_llm = OllamaLLM(model="llama3.1:latest", host="http://localhost:11434")

agent = Agent(
    df_data,
    config={
        "llm": local_llm,
        "enable_code_execution": True,
        "field_descriptions": field_descriptions,
    },
)

# -------------------------
# Query
# -------------------------



class MyREPL:
    def __init__(self, agent):
        # Keep a persistent environment between commands
        self.environment = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,
        }
        self.session = PromptSession() 
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
        else:
            # Non-Python -> forward to the agent
            print("[DEBUG] calling agent.start_new_conversation()")
            self.agent.start_new_conversation()
            print("[DEBUG] calling agent.chat()")
            result = self.agent.chat(expression)
            print("[DEBUG] agent.chat() returned")
            return result

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
