import json
import random

# Load your original file
with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Build a big bag of words from all corpora
all_words = []
for item in data:
    all_words.extend(item["corpus"].split())

def remix_corpus(target_len, exclude_words=None):
    """Create a pseudo-random corpus with roughly target_len words
    using words from all other corpora."""
    words_pool = all_words.copy()
    if exclude_words:
        exclude_set = set(exclude_words)
        words_pool = [w for w in words_pool if w not in exclude_set] or all_words

    return " ".join(random.choice(words_pool) for _ in range(target_len))

new_data = []
for item in data:
    original_words = item["corpus"].split()
    length = len(original_words)

    new_item = {
        "title": item["title"],  # or change the title if you want
        "original_corpus": item["corpus"],
        "remixed_corpus": remix_corpus(length, exclude_words=original_words)
    }
    new_data.append(new_item)

# Save augmented dataset
with open("corpus_remixed.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
