# Civera - Mass Courts Data Pipeline

## EDA
- [EDA](../EDA/README.md)

## Methods

![Prediction Pipeline](images/Civera%20Label%20Prediction%20Pipeline.png)

- [Vector Database Search](../vector_db_search/README.md)
- [Finetune Bert](../bert_finetune/README.md)
- [Finetune LLM](../finetune_llm/README.md)

## How to run

Set thresholds in [cfg/cfg.py](cfg/cfg.py) to combine the three methods.

### *1. Prepare*

#### Extract labels
```
python count.py
```

#### Sample data
```
python sample_data.py
```

### *2. Train*

#### Generate Faiss vector database 
```
python embed.py
```
- Faiss settings are in [db_faiss.py](db_faiss.py)

#### Finetune bert
```
python finetune_bert.py
```

### Finetune LLM
```
python finetune_llm.py
```


### *3. Predict*
#### Predict and write to database
```
python write_to_database.py
```
- Predictor logics are in [predict.py](predict.py)
  - VDBPredictor: [vector_db_search.py](vector_db_search.py)
  - BertPredictor: [bert_inference.py](bert_inference.py)
  - LLMPredictor: [llm_inference.py](llm_inference.py)
