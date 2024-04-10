import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from config import connect_to_mysql, execute_query

class MyDataset(Dataset):
    def __init__(self, connection, query, tokenizer, model_name, method):
        self.connection = connection
        self.query = query
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.method = method
        self.label_dict = {'positive': 0, 'negative': 1}  
        self.data = self.fetch_data()

    def fetch_data(self):
        try:
            records = execute_query(self.connection, self.query)
            dataset = []
            for record in records:
                actor = record['actor'] 
                action = record['action'] 
                description = record['description']  
                label_id = self.label_dict.get(record['label'], -1)  
                text = f"{actor} {action} {description}"
                tokens = text.lower().split(' ')
                dataset.append((tokens, label_id))
            return dataset
        except Exception as e:
            print(f"Error fetching data from MySQL: {e}")
            return []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)

def load_data_from_mysql(query, tokenizer, train_batch_size, test_batch_size, model_name, method, workers, label_dict):
    try:
        connection = connect_to_mysql(host='masscourtsplus.org', port=3306, database='wp_courtdocs', user='buspark', password='U@5p1r3!') 
        if connection is not None:
            train_dataloader = DataLoader(MyDataset(connection, query, tokenizer, model_name, method),
                                          train_batch_size,
                                          shuffle=True,
                                          num_workers=workers,
                                          collate_fn=partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict)),
                                          pin_memory=True)
            test_dataloader = DataLoader(MyDataset(connection, query, tokenizer, model_name, method),
                                         test_batch_size,
                                         shuffle=False,
                                         num_workers=workers,
                                         collate_fn=partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict)),
                                         pin_memory=True)
            return train_dataloader, test_dataloader
        else:
            raise ValueError("Failed to establish connection to MySQL database")
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return None, None
