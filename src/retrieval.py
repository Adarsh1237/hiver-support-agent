"""
Retrieves the most similar historically-resolved customer messages (and their
agent replies) for a new incoming message, to ground the drafted reply.

DECISION: TF-IDF cosine similarity instead of a vector DB / embedding API.
For a few-thousand-row brand subsample this is just as effective, needs no
network call, and is instant to reproduce. Swap-in point for embeddings is
marked below if the real dataset makes TF-IDF too sparse. See decision_log.md #5.
"""
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import ResolvedPair


class Retriever:
    def __init__(self, pairs: List[ResolvedPair]):
        self.pairs = pairs
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        corpus = [p.customer_text for p in pairs]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def top_k(self, message: str, k: int = 3, exclude_tweet_id: str = None) -> List[ResolvedPair]:
        if self.matrix is None or not self.pairs:
            return []
        query_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(
            ((score, i) for i, score in enumerate(sims)),
            reverse=True,
        )
        results = []
        for score, i in ranked:
            pair = self.pairs[i]
            if exclude_tweet_id and pair.tweet_id == exclude_tweet_id:
                continue
            if score <= 0:
                continue
            results.append(pair)
            if len(results) >= k:
                break
        return results
