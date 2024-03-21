## Sample Data
```
python sample.py
```
At present, we sample data based on the action type. We sample at most {LIMIT} rows per action type.
```
query = f"""
           SELECT case_id, actor, action, description
           FROM {table_name}
           WHERE action = "{action_type}"
           LIMIT {_limit};
       """
```
We will further experiment on "sampled_data_based_on_actions_limit=10000.json"

## Embedding
Temporally use "all-MiniLM-L6-v2", may try other options later

## Vector database
Temporally use FAISS (Facebook AI Similarity Search)  

## DB search
- Calculate the embeddings of labeled descriptions
- Store them into a faiss index
- For every unlabeled description
  - calculate the embedding 
  - do similarity search
  - assign labels or create new labels based on the similarity threshold

![db search example](images/vector%20db%20search%20exp.jpg)

