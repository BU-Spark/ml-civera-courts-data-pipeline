# Civera - Mass Courts Data Pipeline

## Methods
#### 1. Vector Database Search
#### 2. Fine-tune Bert
#### 3. Fine-tune LLM

## Steps

#### Extract labels
```
python count.py
```

#### Sample data
```
python sample_data.py
```

#### Finetune bert
```
python finetune_bert.py
```

#### Generate Faiss vector database 
```
python embed.py
```
- Faiss settings are in [db_faiss.py](db_faiss.py)

#### Predict and write to database
```
python write_to_database.py
```
- Predictor logics are in [predict.py](predict.py)
