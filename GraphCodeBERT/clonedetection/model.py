import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 4)
        print("Model Architecture Initialized:")
        print("Dense layer input size:", config.hidden_size*2)
        print("Dense layer output size:", config.hidden_size)
        print("Dropout probability:", config.hidden_dropout_prob)
        print("Output projection size:", 4)

    def forward(self, features, **kwargs):
        print("Features input to classification head:", features.shape)
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        print("Selected <s> token:", x.shape)
        x = x.reshape(-1, x.size(-1)*2)
        print("Reshaped x:", x.shape)
        x = self.dropout(x)
        print("After dropout:", x.shape)
        x = self.dense(x)
        print("After dense layer:", x.shape)
        x = torch.tanh(x)
        print("After tanh activation:", x.shape)
        x = self.dropout(x)
        print("After dropout:", x.shape)
        x = self.out_proj(x)
        print("Output logits:", x.shape)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        # print("Input ids 1:", inputs_ids_1)
        # print("Input ids 2:", inputs_ids_2)
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        # print("Concatenated input ids:", inputs_ids)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        # print("Concatenated position idx:", position_idx)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)
        # print("Concatenated attention mask:", attn_mask)

        # Embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        # print("Input embeddings:", inputs_embeddings)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings * nodes_mask[:,:,None]
        # print("Adjusted input embeddings:", inputs_embeddings)
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]
        # print("Encoder outputs:", outputs)
        logits = self.classifier(outputs)
        # print("Logits:", logits)
        prob = F.softmax(logits, dim=-1)
        # print("Probabilities:", prob)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # print("Loss:", loss)
            return loss, prob
        else:
            return prob

      
        

       
