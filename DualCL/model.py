import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        actor_action_feats = hiddens[:, 0, :]
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(actor_action_feats))
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', actor_action_feats, label_feats)
        outputs = {
            'predicts': predicts,
            'actor_action_feats': actor_action_feats,
            'label_feats': label_feats
        }
        return outputs
