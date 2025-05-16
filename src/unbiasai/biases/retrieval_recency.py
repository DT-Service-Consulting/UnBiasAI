models = ["gpt",  "claude", "mistral", "cohere", "deepseek"]
# retrieval_results structure: { model_name: { query: [list of document results] } }
retrieval_results = {model: {} for model in models}

for model in models:
    for query in test_queries:
        # Set re_rank=True if you wish to re-rank documents using the LLM.
        retrieval_results[model][query] = retrieve(query, model, k=4, re_rank=True)
print("Retrieval complete for all models and queries.")