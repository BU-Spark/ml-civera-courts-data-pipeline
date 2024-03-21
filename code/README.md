## Sample Data
Find more details in [EDA_README](../EDA/README.md)
```
python sample.py
```
Experiment on "sampled_data_limit={LIMIT}.json"

structure:
```
{
  case_action_id: {
    action:
    actor:
    description:
  },
  ...
}
```

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

