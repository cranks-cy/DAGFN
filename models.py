
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GAL

import warnings
warnings.filterwarnings("ignore")

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2= GraphConvolution(nhid, nhid2)

        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj)) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x 
    

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class BaseModel(nn.Module):
    def __init__(self, num_class, fea_size, proj_dim, hidden_dims, num_layers, device, dropout=0.0):
    
        super(BaseModel, self).__init__()

        self.num_class = num_class
        self.fea_size = fea_size

        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.num_layers = num_layers 
        self.dropout = dropout
        self.device = device


        self.GCN1 = GCN(fea_size, self.hidden_dims[0], self.hidden_dims[1], self.dropout)

        self.attention = Attention(self.hidden_dims[1])

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dims[1], 16),
            nn.Tanh(),
            nn.Linear(16, self.num_class),
            nn.LogSoftmax(dim=1)
        )

        self.fc1 = torch.nn.Linear(self.hidden_dims[1], self.proj_dim)
        self.fc2 = torch.nn.Linear(self.proj_dim, self.hidden_dims[1])
    
    def Mlp(self, x):
        return self.MLP(x)

    def Attention(self, x):
        return self.attention(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
   

    def contrastive_loss(self, origin_features, aug_features, same_node_indices = None, temperature=0.6, b_cos: bool = True):
        if same_node_indices is not None and len(same_node_indices) > 0:
            origin_same_features = origin_features[same_node_indices] 
            same_node_embeddings = aug_features[same_node_indices]  
        else:
            origin_same_features = origin_features
            same_node_embeddings = aug_features

        if b_cos:  
            view1, view2 = F.normalize(origin_same_features, dim=1), F.normalize(same_node_embeddings, dim=1) 

        
    
        pos_score = (view1 @ view2.T) / temperature  
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        loss = -score.mean()

        return loss


    def forward(self, features, adj):
    
        inputs = features
        emb = self.GCN1(inputs, adj)
        return emb
       

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        
        labels = torch.argmax(labels, dim=1)
        loss = F.nll_loss(logits, labels, reduction='none')
      
        mask = mask.float()
        mask /= mask.mean()
       
        loss *= mask
      
        return loss.mean()

