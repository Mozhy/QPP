# calculate Primary NQC
results = pd.DataFrame(columns=['qid','score'])
for i,row in dev_queries.iterrows():
  qid = row['id']
  retrieved_list = bm25scores.loc[bm25scores['qid'] == qid]
  nqc_score = 0
  score_D = ScoreCorpus[f"{qid}"]
    
  mean_score = mean(retrieved_list['score'])
  for index, row1 in retrieved_list.iterrows():
    score_d = row1['score']
    nqc_score += pow((score_d - mean_score),2)
    
  nqc_score = nqc_score / len(retrieved_list)
  nqc_score = math.sqrt(nqc_score)
    
  nqc_score = nqc_score / score_D
    
  new_data = pd.DataFrame([[ qid , nqc_score ]],columns= results.columns)
  results = pd.concat([results, new_data],ignore_index=True)

results.to_csv('Dev/nqc_scores_base.txt', header=None, index=None, sep='\t', mode='w')
