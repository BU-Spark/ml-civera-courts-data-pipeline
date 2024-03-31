# Proof of Concept (PoC)

## Project Overview
- **Title**: Legal Case Action/Actor Label Prediction for Mass Court Data Pipeline
- **Date**: 2024/03/30

## Introduction
In this PoC, we aim to demonstrate the effectiveness of our core algorithms and validate the underlying intuition behind our approach to predicting actions and actors in legal cases using a mass court data pipeline. This process involves a combination of natural language processing (NLP) techniques and large language models (LLMs) to enhance the accuracy of our predictions.

## Technologies and Tools
- **Python**: For scripting and automation of the data pipeline.
- **Vector Database**: Utilizing a vector database like FAISS for efficient similarity search in high-dimensional spaces.
- **PyTorch**: For developing and training machine learning models.
- **LLM (Large Language Models)**: To classify or generate new labels for unlabeled data through advanced NLP techniques.

## Implementation Steps
1. **Sample Data Collection**: Gather a representative dataset of labeled legal cases.
2. **Data Embedding**: Use models such as "all-MiniLM-L6-v2" to embed labeled data, transforming text into high-dimensional vectors.
3. **Vector Database Storage**: Store these embeddings in a vector database like FAISS, facilitating efficient retrieval.
4. **Similarity Search for Labeling**: Implement a similarity search in the vector database to find the top K related labeled examples for any given unlabeled data.
5. **Label Prediction with LLMs**: Design and utilize prompts for LLMs to either classify unlabeled data into existing categories or generate new labels as needed.

## Intuition Behind Implementation Steps
- **Sample Data Collection**: Essential for initial testing and adaptation, given the vast 70M row database. Starts with manageable datasets to refine the model.
- **Text Embedding**: Transforms complex legal texts into semantic vectors, crucial for prediction accuracy and facilitates ongoing data integration.
- **Storing Embeddings in Vector Database (FAISS)**: Enhances search efficiency within the extensive legal case database, ensuring swift and accurate retrieval.
- **TopK Similarity Search for Unlabeled Data**: Employs similarity to infer labels for new cases, leveraging patterns found in labeled counterparts.
- **LLM-based Classification or Label Generation**: Applies LLMs' deep contextual understanding to refine label accuracy, especially beneficial for novel or ambiguous cases.

## Potential Challenges
- **Computational Resources**: Large language models require significant computational power, which could present a bottleneck.
- **Cost Considerations**: Depending on the scale of data and frequency of LLM usage, API calls to these models might incur substantial costs.

## How to run
- Set up the environment
```
pip install -r requirements
```
Go to directory src/ (or vector_db_search/)
- Embed and store into vector database
```
python embed.py
```
- Test vector_database similarity search performance
```
python vector_db_search.py
```
- Test LLM prediction performance
  - Temporarily, we experiment with ChatGLM3-6B.
```
python llm.py
```

