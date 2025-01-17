# Calculate Clustered WIG
results = pd.DataFrame(columns=['qid','score'])
for i,row in dev_queries.iterrows():
    qid = row['id']
    query = row['query']
    tokenized_query = query.split(" ")
    query_len = 1 / math.sqrt(len(tokenized_query))
    
    retrieved_list = all_result.loc[all_result['qid'] == qid]
    wig_score = 0
    score_D = ScoreCorpus[f"{qid}"]
    
    res = next((item for item in clustered_data if item["qid"] == qid), None)
    lables = res["lables"]
    medoids_indices = res["medoids_indices"]
   
    df = pd.DataFrame({'qid': retrieved_list["qid"], 'docid': retrieved_list["docid"], 'score': retrieved_list["score"],'cluster': lables})
   
    for g, data in df.groupby('cluster'):
        best_docs = data.head(1)
        for j,row1 in best_docs.iterrows():
            best_doc_score = row1["score"]
            wig_score += (best_doc_score - score_D)

    k = len(medoids_indices) * len(best_docs)
    wig_score = wig_score / k * query_len
        
    new_data = pd.DataFrame([[ qid , wig_score ]],columns= results.columns)
    results = pd.concat([results, new_data],ignore_index=True)

results.to_csv('Dev/K-means/wig_scores/51.txt', header=None, index=None, sep='\t', mode='w')
