# Calculate Clustered SMV
results = pd.DataFrame(columns=['qid','score'])
for i,row in dev_queries.iterrows():
    qid = row['id']
    retrieved_list = all_result.loc[all_result['qid'] == qid]
    
    res = next((item for item in clustered_data if item["qid"] == qid), None)
    lables = res["lables"]
    medoids_indices = res["medoids_indices"]

    smv_score = 0
    score_D = ScoreCorpus[f"{qid}"]
    df = pd.DataFrame({'qid': retrieved_list["qid"], 'docid': retrieved_list["docid"], 'score': retrieved_list["score"],'cluster': lables})
   
    for g, data in df.groupby('cluster'):
        medoid_doc = retrieved_list['score'].iloc[medoids_indices[g]]
        best_docs = data.head(1)
        for j,row1 in best_docs.iterrows():
            best_doc_score = row1["score"]
            w = abs(np.log(best_doc_score / medoid_doc))
            smv_score += best_doc_score * w

    k = len(medoids_indices) * len(best_docs)
    smv_score = smv_score / k
    smv_score = smv_score / score_D
        
    new_data = pd.DataFrame([[ qid , smv_score ]],columns= results.columns)
    results = pd.concat([results, new_data],ignore_index=True)

results.to_csv('Dev/K-means/smv_scores/51.txt', header=None, index=None, sep='\t', mode='w')
