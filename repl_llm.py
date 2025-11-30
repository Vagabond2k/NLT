import pandas as pd
import ollama

from pandasai import Agent
from pandasai.llm.base import LLM
from pandasai.core.prompts.base import BasePrompt
import ast


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
        IMPORTANT RULES (MUST FOLLOW ALL):

        1. Never modify the original dataframe in-place.
        - If the dataframe is named df, do NOT assign to df.
        - Instead, write: df_work = df.copy()
        2. Never reassign df to a filtered version.
        - BAD:  df = df[df["smoker"] == 1]
        - GOOD: df_work = df[df["smoker"] == 1]
        3. Do not keep state in global variables across questions.
        4. Always set the final answer in a variable named `result` as:
        result = {"type": "<type>", "value": <value>}
        5. Do NOT use print() for the final answer.
        """

        prompt = guard + "\n\n" + prompt

        if context is not None and getattr(context, "memory", None) is not None:
            prompt = self.prepend_system_prompt(prompt, context.memory)

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
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


def is_python_code(s: str) -> bool:
    try:
        ast.parse(s)
        return True
    except SyntaxError:
        return False
# -------------------------
# Query
# -------------------------
cimport sys

class MyREPL:
    def __init__(self, agent):
        # Keep a persistent environment between commands
        self.environment = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,
        }
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
            self.agent.start_new_conversation()
            return self.agent.chat(expression)

    def run(self):
        print("My Custom REPL - type 'exit' or 'quit' (or Ctrl+D) to quit")

        while True:
            try:
                user_input = input(">> ")

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
                # Send errors to stderr so they’re visually distinct
                print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":

    repl = MyREPL(agent)
    repl.run()
