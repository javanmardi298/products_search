import json
import math
import os
import random

import numpy as np
import pandas as pd
import requests

URL = "https://github.com/wayfair/WANDS/raw/main/dataset/"
SEED = 11


def download(name, folder="data/wands"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print("downloading", name)
        r = requests.get(URL + name)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def load_data():
    products = download("product.csv")
    queries = download("query.csv")
    labels = download("label.csv")

    for c in ["product_name", "product_class", "category_hierarchy", "product_description", "product_features"]:
        products[c] = products[c].fillna("").astype(str)

    score = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
    qrels = {}
    for q, p, label in zip(labels["query_id"], labels["product_id"], labels["label"]):
        q, p = int(q), int(p)
        if q not in qrels:
            qrels[q] = {}
        qrels[q][p] = score[label]

    query_text = dict(zip(queries["query_id"].astype(int), queries["query"]))
    return products, queries, labels, qrels, query_text


def features_to_text(raw, max_n=15):
    out = []
    for part in raw.split("|"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        if val.strip():
            out.append(key.strip() + ": " + val.strip())
    return "; ".join(out[:max_n])


def product_text(p, rich):
    if not rich:
        return p.product_name + ". " + p.product_description[:1000]
    return (p.product_name + "\n"
            + "Category: " + p.product_class + " (" + p.category_hierarchy + ")\n"
            + "Features: " + features_to_text(p.product_features) + "\n"
            + "Description: " + p.product_description[:600])


def make_texts(products):
    basic, rich = [], []
    for p in products.itertuples():
        basic.append(product_text(p, rich=False))
        rich.append(product_text(p, rich=True))
    return {"basic": basic, "rich": rich}


def make_split(qrels, n_dev=160):
    if os.path.exists("split.json"):
        with open("split.json") as f:
            return json.load(f)
    ids = sorted(qrels.keys())
    random.Random(SEED).shuffle(ids)
    split = {"dev": sorted(ids[:n_dev]), "test": sorted(ids[n_dev:])}
    with open("split.json", "w") as f:
        json.dump(split, f)
    return split


def dcg(gains):
    total = 0
    for i, g in enumerate(gains):
        total += (2 ** g - 1) / math.log2(i + 2)
    return total


def ndcg(ranked, rel, k=10):
    best = dcg(sorted(rel.values(), reverse=True)[:k])
    if best == 0:
        return float("nan")
    return dcg([rel.get(p, 0) for p in ranked[:k]]) / best


def recall(ranked, rel, k=10):
    exact = [p for p in rel if rel[p] == 2]
    if not exact:
        return float("nan")
    found = [p for p in ranked[:k] if p in exact]
    return len(found) / len(exact)


def mrr(ranked, rel):
    for i, p in enumerate(ranked):
        if rel.get(p) == 2:
            return 1 / (i + 1)
    return 0


def judged(ranked, rel, k=10):
    # share of the top k that has a label at all
    n = 0
    for p in ranked[:k]:
        if p in rel:
            n += 1
    return n / k


def evaluate(run, qrels):
    rows = []
    for q, ranked in run.items():
        rel = qrels[q]
        rows.append({"qid": q, "ndcg@10": ndcg(ranked, rel), "recall@10": recall(ranked, rel),
                     "mrr": mrr(ranked, rel), "judged@10": judged(ranked, rel)})
    return pd.DataFrame(rows).set_index("qid")


def bootstrap(a, b, n=10000):
    # is b better than a on the same queries?
    diff = (b - a).dropna().values
    rng = np.random.default_rng(SEED)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"diff": round(diff.mean(), 3), "low": round(low, 3), "high": round(high, 3),
            "p": round((means <= 0).mean(), 4)}


def short_name(model):
    return model.split("/")[-1].replace(":", "-")
