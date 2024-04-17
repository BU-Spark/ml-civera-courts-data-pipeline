import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
import mysql.connector

class MyDataset(Dataset):
    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = []
        for data in raw_data:
            tokens = data['description'].lower().split(' ')
            label_id = label_dict.get(data.get('actor_action'), -1)  
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        tokens, label_id = self._dataset[index]
        encoded_tokens = self.tokenizer(tokens, padding=True, truncation=True, max_length=256,
                                        is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
        return encoded_tokens, torch.tensor(label_id)

    def __len__(self):
        return len(self._dataset)

def my_collate(batch, method, num_classes):
    tokens, label_ids = zip(*batch)
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(tokens[0]['input_ids'])[:, num_classes:]
        positions += torch.arange(0, tokens[0]['input_ids'].size(1) - num_classes)
        tokens[0]['position_ids'] = positions
    return tokens, torch.tensor(label_ids)

def create_label_dict(raw_data):
    label_dict = {}
    descriptions = []
    for data in raw_data:
        actor = data.get('actor')
        action = data.get('action')
        description = data.get('description')
        if actor and action:
            label = f"{actor}_{action}"
        elif action:
            label = action
        elif actor:
            label = actor
        else:
            continue
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        if description:  
            descriptions.append(description)
    num_classes = len(label_dict)
    return label_dict, descriptions, num_classes

def establish_database_connection(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def close_database_connection(connection):
    if connection is not None and connection.is_connected():
        connection.close()

def load_data_from_mysql(connection, tokenizer, model_name, method, train_batch_size, test_batch_size, workers):
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT actor, action, description
            FROM cdocs_case_action_index
            TABLESAMPLE BERNOULLI(1)
        """)
        raw_data = cursor.fetchall()
        
        label_dict = create_label_dict(raw_data)  

        trainset = MyDataset(raw_data, label_dict, tokenizer, model_name, method)
        testset = MyDataset(raw_data, label_dict, tokenizer, model_name, method)

        collate_fn = partial(my_collate, method=method, num_classes=len(label_dict))
        train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
        test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)

        return train_dataloader, test_dataloader
    except mysql.connector.Error as e:
        print(f"Error loading data from MySQL: {e}")
        return None, None
    finally:
        close_database_connection(connection)
