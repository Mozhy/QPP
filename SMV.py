# calculate Primary SMV
results = pd.DataFrame(columns=['qid','score'])
for i,row in dev_queries.iterrows():
  qid = row['id']
  retrieved_list = bm25scores.loc[bm25scores['qid'] == qid]
  k = len(retrieved_list)
  score_D = ScoreCorpus[f"{qid}"]
  smv_score = 0

  mean_score = mean(retrieved_list['score'])
  for index, row1 in retrieved_list.iterrows():
      doc_score = row1['score']
      w = abs(np.log(doc_score / mean_score))
      smv_score += doc_score * w

  smv_score = smv_score / k
  smv_score = smv_score / score_D

  new_data = pd.DataFrame([[ qid , smv_score ]],columns= results.columns)
  results = pd.concat([results, new_data],ignore_index=True)

results.to_csv('Dev/smv_scores_base.txt', header=None, index=None, sep='\t', mode='w')
