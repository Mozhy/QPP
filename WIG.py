# calculate Primary WIG
results = pd.DataFrame(columns=['qid','score'])
for i,row in dev_queries.iterrows():
  qid = row['id']
  query = row['query']
  tokenized_query = query.split(" ")
  query_len = 1 / math.sqrt(len(tokenized_query))

  retrieved_list = bm25scores.loc[bm25scores['qid'] == qid]
  wig_score = 0
  score_D = ScoreCorpus[f"{qid}"]
    
  for index, row1 in retrieved_list.iterrows():
      score_d = row1['score']
      wig_score += (score_d - score_D)

  wig_score = wig_score / len(retrieved_list) * query_len
  new_data = pd.DataFrame([[ qid , wig_score ]],columns= results.columns)
  results = pd.concat([results, new_data],ignore_index=True)

results.to_csv('Dev/wig_scores_base.txt', header=None, index=None, sep='\t', mode='w')
