from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the model
llm = ChatOllama(
    model="gemma2:9b",   # Must match the name shown by `ollama list`
    temperature=0.2,
    num_ctx=8192,
    # base_url="http://localhost:11434",  # Uncomment/change if Ollama runs elsewhere
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "{question}"),
])

# Request text
request = (
    "Generate 5 fictional newspaper-style articles in English, each between 400 and 700 words.\n\n"
    "Each article should include several Named Entities, such as fictional people, organizations, "
    "locations, and dates.\n\n"
    "The articles should be realistic in tone and style but completely fictional — do not use or "
    "reference real events, places, or people.\n\n"
    "Format the output strictly as valid JSON, structured as an array of 5 objects.\n"
    "Each object must have this structure:\n"
    "{\n"
    '  "title": "example title",\n'
    '  "corpus": "example article text"\n'
    "}\n"
    "Only return the JSON — no explanations."
)

# Build chain and parse plain text
chain = prompt | llm | StrOutputParser()

# Invoke and print
print(chain.invoke({"question": request}))