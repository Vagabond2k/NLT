# Python Requirements
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Ollama Requirements
https://ollama.com/library/llama3.1

## Model to use in local case is a 8Billion one with 4.9 size in ram, which means is quantized in around 4bit as Q4_K_M.
```
ollama pull llama3.1:latest
ollama serve
```

# Part 1
```
python3 langchain_test.py
```
# Part 2 
```
python3 random_corpus.py 
python3 named_entities.py
```
# Part 3 
```
python3 repl_corpus.py
```
# Part 4 example of how to use an Agent, this was for a mini assignment on week 10
```
python3 repl_llm_minimal.py 
```

# Part 4 using PandasAI 
```
python3 repl_llm_pandasai.py
```

# Part 4 not using preconfigured framework
```
python3 repl_llm.py
```

# If you need to increase the RAM 
```
Disable current swap 

sudo swapoff -a 

Resize or recreate the swapfile 

sudo rm /swapfile 

sudo fallocate -l 16G /swapfile 

Secure permissions 

sudo chmod 600 /swapfile 

Make it a swap area 

sudo mkswap /swapfile 

Re-enable it 

sudo swapon /swapfile 
```