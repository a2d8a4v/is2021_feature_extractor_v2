import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(2, 6, 5)
#         self.pool = nn.MaxPool1d(2, 2)
#         self.conv2 = nn.Conv1d(6, 16, 5)
#         # self.fc1 = nn.Linear(272, 120)
#         self.fc1 = nn.Linear(3952, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 50)
#         self.fc4 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, checkpoint, freeze_bert=False, padding=500):
        super(BertClassifier, self).__init__()
        D_in, D_out = 768, 3 # [PAD] is 0

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(checkpoint)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out)
        )

        # issue: BERT model will seperate some unusual words into BPE
        self.conv1 = nn.Conv1d(padding, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, pro_idx, *args, **kwargs):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask, *args, **kwargs)
        # Extract the last hidden state of all the tokens
        # outputs.last_hidden_state[:,0,:] `[CLS]`
        # @https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
        last_hidden_state = outputs.last_hidden_state
        for i, l_ in enumerate(pro_idx):
            if l_: ## array is not empty
                g = last_hidden_state[i,:,:].unsqueeze(0).clone()
                g[:,:,:] = 0
                for e_ in l_[::-1]:
                    a = g.clone()
                    a[:,e_[0]:e_[1]+1,:] = last_hidden_state[i,e_[0]:e_[1]+1,:].clone()
                    s = self.conv1(a)
                    last_hidden_state[i,e_[0],:] = s.clone()
                    ## cut and paste
                    c_1 = last_hidden_state[i,:e_[0]+1,:].clone()
                    if len(c_1.shape) < 3:
                        c_1 = c_1.unsqueeze(0)
                    c_2 = last_hidden_state[i,e_[0]+1:e_[1]+1,:].clone()
                    if len(c_2.shape) < 3:
                        c_2 = c_2.unsqueeze(0)
                    c_3 = last_hidden_state[i,e_[1]+1:-1,:].clone()
                    if len(c_3.shape) < 3:
                        c_3 = c_3.unsqueeze(0)
                    c_4 = last_hidden_state[i,-1:,:].unsqueeze(0) # [SEP]
                    last_hidden_state[i,:,:] = torch.cat((c_1, c_3, c_2, c_4), 1).clone()
        logits = self.classifier(last_hidden_state)

        return logits