from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma2:9b",      # exactly the name you see in `ollama list`
    temperature=0.2,
    num_ctx=8192,           # matches your model’s context
    # base_url="http://localhost:11434",  # change if Ollama runs elsewhere
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "{question}")
])

chain = prompt | llm
print(chain.invoke({"question": "Explain LoRA fine-tuning in 3 bullet points."}).content)