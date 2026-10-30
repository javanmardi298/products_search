from shop_common import build_texts, load_wands, ndcg, recall

products, queries, labels, qrels, query_text = load_wands()
print(f"{len(products):,} products | {len(queries)} queries | {len(labels):,} labels")

print("\nexample product as the index sees it:\n")
print(build_texts(products.head(1))["rich"][0])

rel = {10: 2, 11: 1, 12: 0}
assert ndcg([10, 11, 12], rel) == 1.0
assert ndcg([12, 11, 10], rel) < 1.0
assert recall([10], rel) == 1.0
print("\nmetrics OK")
