import os
import sys
import torch
import logging
import argparse
from datetime import datetime
import mysql.connector

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='cdocs_case_action_index', help='Name of the table in your database')
    parser.add_argument('--model_name', type=str, default='google-bert/bert-base-uncased', help='Type of model architecture')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.01, help='Decay rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for DualCL method')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature parameter for DualCL method')
    parser.add_argument('--backend', default=False, action='store_true', help='Use backend')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    connection = mysql.connector.connect(
        host="masscourtsplus.org",
        user="buspark",
        password="U@5p1r3!",
        database="wp_courtdocs"
    )

    try:    
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT actor, action FROM {args.dataset}")
        results = cursor.fetchall()
        num_classes = len(results)
        args.num_classes = num_classes
    
        log_name = '{}_{}_{}.log'.format(args.dataset, args.model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
        args.log_name = os.path.join(args.log_dir, log_name)
        
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(args.log_name))
        
        return args, logger
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    args, logger = get_config()
