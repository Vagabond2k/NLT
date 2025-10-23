# Can you create a main and a function in python that allow me to interact with my local ollama model ?
# model : gemma 2 9b
# language: python

import llama_cpp

def interact_with_gemma():
  """Interacts with the local Gemma 2 9B model."""
  model_path = "path/to/your/gemma_2_9b_model.bin"  # Replace with your model's path
  llm = llama_cpp.Llama(model_path=model_path)

  while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
      break

    response = llm.generate_text(user_input, max_tokens=100)
    print("Gemma:", response.strip())

if __name__ == "__main__":
  interact_with_gemma()