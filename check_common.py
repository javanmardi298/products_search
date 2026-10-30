from shop_common import load_data, make_texts, ndcg, recall

products, queries, labels, qrels, query_text = load_data()
print(len(products), "products,", len(queries), "queries,", len(labels), "labels")
print()
print(make_texts(products.head(1))["rich"][0])
print()

# small test of the metrics
rel = {10: 2, 11: 1, 12: 0}
print(ndcg([10, 11, 12], rel))  # 1.0
print(ndcg([12, 11, 10], rel))  # less than 1
print(recall([10], rel))        # 1.0
