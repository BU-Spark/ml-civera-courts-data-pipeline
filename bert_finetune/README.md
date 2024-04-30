# Method: Finetune Bert

### Model structure
- Bert + 2 classifiers
- Bert: bert-base-uncased

### How to run
- Finetune
```
python method_bert.py --epoch 5 --limit 1000
```
- Evaluate
```
python method_bert.py --eval
```
