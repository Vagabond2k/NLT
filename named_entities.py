import spacy
import json
from spacy import displacy

with open('corpus.json', 'r') as file:
    data = json.load(file)

nlp = spacy.load("en_core_web_sm")

for item in data:
    doc = nlp(item["corpus"])
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print("in doc: ", item["title"])
    print("there were this many entities: ", len(doc.ents))
