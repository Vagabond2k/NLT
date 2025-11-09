from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from typing import List



class TFidModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.x_test = []
        self.x_train = []
        self.x_new_pred = []

    def vectorize_and_learn(self, corpus):
        self.x_train = self.vectorizer.fit_transform(corpus)  
        print(self.x_train)

    def vectorize(self, corpus):
        self.x_new_pred = self.vectorizer.transform(corpus)
        print(self.x_new_pred)

    def compare(self, titles):
        sim = cosine_similarity(self.x_train, self.x_new_pred)
        df_sim = pd.DataFrame(sim, index=titles, columns=[1])
        return df_sim


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
        self.environment = {}  # Store variables
        self.environment['model'] = TFidModel()
       
   
    def evaluate(self, expression):
        # Your custom evaluation logic here
        # This is where you'd parse and execute your language
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
                self.environment['model'].vectorize([text])
                titles = [article.title for article in self.environment['articles']]
                sim = self.environment['model'].compare(titles)
                return sim
         
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
