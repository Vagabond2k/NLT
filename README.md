python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
#Part 1
python3 langchain_test.py
#Part 2 
python3 random_corpus.py 
python3 named_entities.py
#Part 3 
python3 repl_corpus.py
#Part 4 example of how to use an Agent 
python3 repl_llm_minimal.py 
#Part 4 
python3 repl_llm.py


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