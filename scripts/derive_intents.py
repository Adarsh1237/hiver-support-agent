"""
Clusters a brand's inbound customer messages (TF-IDF + KMeans) and prints
top terms per cluster, so you can sanity-check whether config.py:INTENTS
actually matches the brand's real traffic before committing to it -- or
derive a fresh taxonomy for a different brand.

python -m scripts.derive_intents --brand AmazonHelp --k 7
"""
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src import config
from src.data_loader import load_brand_pairs


def main(brand: str, k: int):
    pairs = load_brand_pairs(brand)
    texts = [p.customer_text for p in pairs]
    if len(texts) < k:
        raise SystemExit(f"Only {len(texts)} messages for brand={brand}, need >= k={k}.")

    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=2000)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

    terms = vectorizer.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_size = sum(1 for lbl in km.labels_ if lbl == i)
        example = next(t for t, lbl in zip(texts, km.labels_) if lbl == i)
        print(f"\nCluster {i} (n={cluster_size})")
        print("  top terms:", ", ".join(top_terms))
        print("  example:  ", example[:120])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=config.DEFAULT_BRAND)
    p.add_argument("--k", type=int, default=len(config.INTENTS))
    args = p.parse_args()
    main(args.brand, args.k)
