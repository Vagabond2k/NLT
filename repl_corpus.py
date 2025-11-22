from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import json
from typing import List
 

class TFidModel:
    def __init__(self, 
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9):
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self.x_test = []
        self.x_train = []
        self.x_new_pred = []

    def vectorize_and_learn(self, corpus):
        self.x_train = self.vectorizer.fit_transform(corpus)  
    
    def vectorize(self, corpus):
        self.x_new_pred = self.vectorizer.transform(corpus)
    
    def compare(self, titles):
        sim = cosine_similarity(self.x_train, self.x_new_pred)
        df_sim = pd.DataFrame(sim, index=titles, columns=['similarity'])
        df_sorted = df_sim.sort_values(by='similarity', ascending=False)
        return df_sorted
    

class BagOfWordModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.x_test = []
        self.x_train = []
        self.x_new_pred = []

    def vectorize_and_learn(self, corpus):
        self.x_train = self.vectorizer.fit_transform(corpus)  
    
    def vectorize(self, corpus):
        self.x_new_pred = self.vectorizer.transform(corpus)
    
    def compare(self, titles):
        sim = cosine_similarity(self.x_train, self.x_new_pred)
        df_sim = pd.DataFrame(sim, index=titles, columns=['similarity'])
        df_sorted = df_sim.sort_values(by='similarity', ascending=False)
        return df_sorted
            

# pretrained_weights: str, default='distilbert-base-uncased' 
# ensemble_method: str, default='average' - How word vectors are aggregated into sentece vectors.    

class EmbeddingsModel:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.x_train = None
        self.x_new_pred = None

    def _reset_vectors(self):
        # best-effort reset; harmless if attribute doesn’t exist
        try:
            self.vectorizer.vectors = []
        except Exception:
            pass

    def vectorize_and_learn(self, corpus):
        self._reset_vectors()
        self.vectorizer.run(corpus)
        self.x_train = np.array(self.vectorizer.vectors)

    def vectorize(self, corpus):
        self._reset_vectors()
        self.vectorizer.run(corpus)
        self.x_new_pred = np.array(self.vectorizer.vectors)

    def compare(self, titles):
        sim = cosine_similarity(self.x_train, self.x_new_pred)
        # if single query, squeeze to 1 column; else name the columns
        if sim.shape[1] == 1:
            df_sim = pd.DataFrame(sim[:, 0], index=titles, columns=['similarity'])
        else:
            cols = [f'query_{i}' for i in range(sim.shape[1])]
            df_sim = pd.DataFrame(sim, index=titles, columns=cols)
        return df_sim.sort_values(by=df_sim.columns[0], ascending=False)


class SentenceEmbeddingsModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # This replaces Vectorizer() from sent2vec
        self.vectorizer = SentenceTransformer(model_name)
        self.x_train = None
        self.x_new_pred = None

    def vectorize_and_learn(self, corpus):
        """
        Encode the training corpus and store it.
        SentenceTransformer.encode:
          - returns one vector per input string
          - we ask it to normalize embeddings so cosine ~= dot product
        """
        self.x_train = self.vectorizer.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True
        )  # shape: (n_docs, dim)

    def vectorize(self, corpus):
        """
        Encode new text(s) to compare against the corpus.
        """
        self.x_new_pred = self.vectorizer.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True
        )  # shape: (n_queries, dim)

    def compare(self, titles):
        """
        Compute cosine similarity between each training doc and each query.

        If you pass a single query (your REPL does that),
        you'll get a column called 'similarity'.
        """
        # Because embeddings are normalized, cosine_similarity and dot product are equivalent.
        sim = cosine_similarity(self.x_train, self.x_new_pred)  # (n_docs, n_queries)

        if sim.shape[1] == 1:
            # one query -> single column
            df_sim = pd.DataFrame(sim[:, 0], index=titles, columns=['similarity'])
        else:
            # multiple queries -> one column per query
            cols = [f'query_{i}' for i in range(sim.shape[1])]
            df_sim = pd.DataFrame(sim, index=titles, columns=cols)

        return df_sim.sort_values(by=df_sim.columns[0], ascending=False)


class Article:
    def __init__(self, title: str, corpus: str):
        self.title = title
        self.corpus = corpus
   
    def __repr__(self):
        return f"Article(title='{self.title[:50]}...', corpus='{self.corpus[:50]}...')"
   
    def __str__(self):
        return f"Title: {self.title}\nCorpus: {self.corpus}"

class ArticleLoader:
    @staticmethod
    def load_articles(file_path: str) -> List[Article]:
        """Load articles from JSON file and return list of Article objects"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
       
        articles = [Article(item['title'], item['corpus']) for item in data]
        return articles
   
    @staticmethod
    def load_as_dict(file_path: str) -> List[dict]:
        """Load articles as list of dictionaries"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

class MyREPL:
    def __init__(self):
        self.environment = {}
        self.environment['encoding'] = 'tfid'
        self.environment['model'] = TFidModel()

    def _model_ready(self, model):
        xt = getattr(model, "x_train", None)
        if xt is None:
            return False
        # Works for numpy arrays and sparse matrices
        if hasattr(xt, "shape"):
            return xt.shape[0] > 0
        try:
            return len(xt) > 0
        except TypeError:
            return False

    def evaluate(self, expression):

        if expression.startswith('Config Encoding'):
            encoding = expression.replace('Config Encoding ', '', 1).strip()
            self.environment['encoding'] = encoding
            if encoding == 'bow':
                self.environment['model'] = BagOfWordModel()
            elif encoding == 'tfid':
                self.environment['model'] = TFidModel()
            elif encoding == 'dist':
                # distilbert-base-uncased
                self.environment['model'] = EmbeddingsModel()
            elif encoding == 'dsent':
                # all-MiniLM-L6-v2
                self.environment['model'] = SentenceEmbeddingsModel()
            else:
                return f"Unknown encoding: {encoding}"
            return 'Config complete!'
        
        if expression.startswith('Stats'):
            # sihluette
            # we have 6 cluster as we have 6 topic
            k = 6
            self.environment['kmeans'] = KMeans(n_clusters=k, random_state=42)
            self.environment['labels'] = self.environment['kmeans'].fit_predict(self.environment['model'].x_train)
            score = silhouette_score(self.environment['model'].x_train, self.environment['labels'], metric="cosine")

            stats = f"k={k}, silhouette={score:.3f}"
            return stats
        
        if expression.startswith('Visual'):
            # --- use the embedding model you clustered with ---
            saved_model = self.environment['model']
            labels = self.environment['labels']
            kmeans = self.environment['kmeans']  # <- make sure you store kmeans here

            # normalize train embeddings
            X_train_norm = normalize(saved_model.x_train)

            # check if we have a query embedding
            has_query = (
                hasattr(saved_model, "x_new_pred")
                and saved_model.x_new_pred is not None
                and saved_model.x_new_pred.shape[0] > 0
            )

            if has_query:
                # normalize query
                query_norm = normalize(saved_model.x_new_pred)
                # stack train + query for t-SNE
                X_norm = np.vstack([X_train_norm, query_norm])
            else:
                X_norm = X_train_norm

            # --- t-SNE on (train [+ query]) ---
            tsne = TSNE(  # t-Distributed Stochastic Neighbor Embedding
                n_components=2,
                metric="cosine",
                perplexity=10,
                random_state=42
            )
            X_tsne_all = tsne.fit_transform(X_norm)

            if has_query:
                X_tsne = X_tsne_all[:-1]      # documents
                query_tsne = X_tsne_all[-1]   # query point
            else:
                X_tsne = X_tsne_all
                query_tsne = None

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=labels,
                alpha=0.7,
                s=20
            )
            plt.title("t-SNE visualization of sentence embeddings (colored by cluster)")
            plt.xlabel("t-SNE dim 1")
            plt.ylabel("t-SNE dim 2")

            # article & label text (unchanged)
            for i, art in enumerate(self.environment['articles']):
                plt.text(
                    X_tsne[i, 0],
                    X_tsne[i, 1],
                    art.title[:20],
                    fontsize=7
                )
                plt.text(
                    X_tsne[i, 0] + 0.5,
                    X_tsne[i, 1] + 0.5,
                    str(labels[i]),
                    fontsize=8
                )

            # --- centroids in t-SNE space ---
            centroids_2d = {}
            for cid in np.unique(labels):
                if cid == -1:
                    continue
                points = X_tsne[labels == cid]
                centroids_2d[cid] = points.mean(axis=0)

            for cid, c in centroids_2d.items():
                plt.scatter(
                    c[0], c[1],
                    marker='X',
                    s=200,
                    edgecolor='black',
                    linewidths=1.5,
                )

            # --- build topic labels using a TF-IDF model ---
            tfidf_model = TFidModel()  # use your improved version with stop_words etc.
            tfidf_model.vectorize_and_learn(
                [article.corpus for article in self.environment['articles']]
            )
            tfidf = tfidf_model.x_train
            terms = tfidf_model.vectorizer.get_feature_names_out()

            topic_labels = {}
            for cid in np.unique(labels):
                idx = np.where(labels == cid)[0]
                tfidf_cluster = tfidf[idx]
                scores = tfidf_cluster.mean(axis=0).A1
                n = 2
                top_idx = scores.argsort()[::-1][:n]
                top_terms = [terms[i] for i in top_idx]
                topic_labels[cid] = top_terms

            self.environment["topic_labels"] = topic_labels

            # draw centroids again with labels
            for cid, c in centroids_2d.items():
                plt.scatter(
                    c[0], c[1],
                    marker='X',
                    s=200,
                    edgecolor='black',
                    linewidths=1.5,
                )
                label = ", ".join(self.environment["topic_labels"].get(cid, [f"C{cid}"]))
                plt.text(
                    c[0] + 0.3,
                    c[1] + 0.3,
                    label,
                    fontsize=10,
                    fontweight='bold'
                )

            # --- plot query point & cluster if we have a query ---
            if has_query:
                # cosine similarity query vs cluster centers (in embedding space)
                cluster_centers = kmeans.cluster_centers_
                cluster_centers_norm = normalize(cluster_centers)
                sims = cosine_similarity(query_norm, cluster_centers_norm)[0]
                query_cluster = int(sims.argmax())

                plt.scatter(
                    query_tsne[0],
                    query_tsne[1],
                    marker='*',
                    s=300,
                    edgecolor='red',
                    linewidths=2,
                )
                plt.text(
                    query_tsne[0] + 0.5,
                    query_tsne[1] + 0.5,
                    f"QUERY (cluster {query_cluster})",
                    fontsize=10,
                    fontweight='bold'
                )

            # restore original model object
            self.environment['model'] = saved_model

            plt.colorbar(scatter, label="Cluster ID")
            plt.tight_layout()
            plt.show()
          
        if expression.startswith('Load Corpus'):
            filename = expression.replace('Load Corpus ', '', 1).strip()
            if filename:
                articles = ArticleLoader.load_articles(filename)
                self.environment['articles'] = articles
                corpus = [article.corpus for article in articles]
                self.environment['model'].vectorize_and_learn(corpus)
                return articles

        if expression.startswith('Compare text'):
            text = expression.split("Compare text ", 1)[1].split('\n', 1)[0].strip()
            if text:
                # must have articles loaded
                articles = self.environment.get('articles')
                if not articles:
                    return "No corpus loaded. Run 'Load Corpus <file>' first."

                model = self.environment['model']

                # If model was re-initialized (e.g., after Config Encoding), rebuild x_train now
                if not self._model_ready(model):
                    corpus = [a.corpus for a in articles]
                    model.vectorize_and_learn(corpus)

                # Encode query and compare
                model.vectorize([text])
                titles = [a.title for a in articles]
                return model.compare(titles)

        return eval(expression, self.environment)
   
    def run(self):
        print("My Custom REPL - type 'exit' to quit")
       
        while True:
            try:
                user_input = input(">> ")
               
                if user_input.strip().lower() == 'exit':
                    break
               
                if not user_input.strip():
                    continue
               
                result = self.evaluate(user_input)
               
                if result is not None:
                    if (isinstance(result, pd.DataFrame)):
                        print(result.round(2))
                    else:
                        print(result)
                   
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    repl = MyREPL()
    repl.run()
