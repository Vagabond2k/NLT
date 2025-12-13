# Python Requirements 3.12
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

# Python Requirements 3.11 In case to test the PandasAI version of the Agent.
you can see here pandasAI only works on python < 3.11.9
Also you have to change and use a dedicated requirements file named requirements_pandasai.txt
```
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

exec $SHELL
pyenv install 3.11.9
pyenv local 3.11.9
pip install -r requirements_pandasai.txt
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
python3 named_entities_improve.py
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

# Part 5 ReACT Agent with COT memory and summarization
```
python3 multi_tools.py
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