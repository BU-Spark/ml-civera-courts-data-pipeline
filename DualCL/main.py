import torch
import sys
import os
from tqdm import tqdm
from config import get_config
from data import load_data_from_mysql
from model import Transformer
from loss import CELoss, SupConLoss, DualLoss
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

class Instructor:
    def __init__(self, args, logger, connection):
        self.args = args
        self.logger = logger
        self.connection = connection
        
        if args.model_name == 'google-bert/bert-base-uncased':
            self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=True)
            self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased", use_auth_token=True)
        else:
            raise ValueError('Unknown model name')

        base_model = AutoModel.from_pretrained(args.model_name)
        self.model = Transformer(base_model, args.num_classes, args.method).to(args.device)
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()

        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss = 0
        correct_predictions = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                _, predicted_labels = torch.max(outputs['predicts'], 1)
                correct_predictions += (predicted_labels == targets).sum().item()
                total_samples += targets.size(0)

        average_loss = test_loss / total_samples
        accuracy = correct_predictions / total_samples

        return average_loss, accuracy

    def run(self):
        try:
            train_dataloader, test_dataloader = load_data_from_mysql(
                query="""SELECT actor, action, description
                        FROM cdocs_case_action_index
                        LIMIT 10000""",
                tokenizer=self.tokenizer,
                model_name=self.args.model_name,
                method=self.args.method,
                train_batch_size=self.args.train_batch_size,
                test_batch_size=self.args.test_batch_size,
                workers=0
            )

            _params = filter(lambda p: p.requires_grad, self.model.parameters())

            if self.args.method == 'ce':
                criterion = CELoss()
            elif self.args.method == 'scl':
                criterion = SupConLoss(self.args.alpha, self.args.temp)
            elif self.args.method == 'dualcl':
                criterion = DualLoss(self.args.alpha, self.args.temp)
            else:
                raise ValueError('Unknown method')

            optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
            best_loss = float('inf')
            best_acc = 0

            for epoch in range(self.args.num_epoch):
                train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
                test_loss, test_acc = self._test(test_dataloader, criterion)
                self.logger.info(f'Epoch {epoch + 1}/{self.args.num_epoch}')
                self.logger.info(f'[Train] Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%')
                self.logger.info(f'[Test] Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%')

                if test_loss < best_loss:
                    best_loss = test_loss
                if test_acc > best_acc:
                    best_acc = test_acc
                
            self.logger.info(f'Best loss: {best_loss:.4f}, Best accuracy: {best_acc * 100:.2f}%')
            self.logger.info(f'Log saved: {self.args.log_name}')

        except Exception as e:
            self.logger.error(f'An error occurred: {e}')
            sys.exit(1)

def save_model(model, save_dir='~/Documents/saved_models', filename='model.pth'):
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved at: {filepath}")

if __name__ == '__main__':
    args, logger, connection = get_config()
    ins = Instructor(args, logger, connection)
    ins.run()
