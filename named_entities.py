import spacy
import json
from spacy import displacy
# TD-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

with open('corpus.json', 'r') as file:
    data = json.load(file)

nlp = spacy.load("en_core_web_sm")

for item in data:
    doc = nlp(item["corpus"])
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print("in doc: ", item["title"])
    print("there were this many entities: ", len(doc.ents))

corpus = [item["corpus"] for item in data]
titles = [item["title"] for item in data]  



# TF-IDF (tweak params as needed)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)  # shape: (n_docs, n_terms)


# Use sparse DataFrame to save memory; labels must match number of docs
labels = titles  # or [f"doc{i+1}" for i in range(len(corpus))]
df_tfidf = pd.DataFrame.sparse.from_spmatrix(
    X, index=labels, columns=vectorizer.get_feature_names_out()
)

print("This is the TF-IDF style representation in a dataframe format")
print(df_tfidf.head(30))

# Cosine similarity (works directly on the sparse matrix X)
sim = cosine_similarity(X, X)

# Put similarity into a DataFrame with the same labels
df_sim = pd.DataFrame(sim, index=labels, columns=labels)
print("Cosine similarity matrix:")
print(df_sim.round(3))

#get me the most releveant terms per doc
def top_k_terms_for_doc(row_idx, k=10):
    row = X[row_idx].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    idx = row.argsort()[::-1][:k]
    return [(terms[i], float(row[i])) for i in idx if row[i] > 0]

for i, title in enumerate(df_tfidf.index):
    print(f"\n📰 {title}")
    for term, weight in top_k_terms_for_doc(i, k=10):
        print(f"  - {term}: {weight:.3f}")
