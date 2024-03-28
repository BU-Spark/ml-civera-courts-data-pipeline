# Civera
The Civera Project for CS549  
Group members:  
Boyang Liu, liubyfly@bu.edu   
Christopher Ziko  
Jamie Lin  
Jeffrey Zhou  

----

## EDA

[EDA_README](EDA/README.md)

---

## Method and Experiment

### *Preprocessed data*
We experiment on "EDA/sample/sampled_data_limit={_LIMIT}.json"

Structure:
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

### *How to design and implement methods*
- Make sure there's "sampled_data_limit={_limit}.json" file in EDA/sample 
  - Otherwise, follow "How to run" in [EDA_README](../EDA/README.md)
- Copy "methodTemplate" to "YourMethodName"
- Go to YourMethodName/
```
python method_template.py
```

### Vector DB Search
[vector_db_search_README](vector_db_search/README.md)
