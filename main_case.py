from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import random
import pandas as pd
import glob

from utils_case import precision_recall_f1, accuracy, ft_load_graph, load_data, calculate_specificity_and_negative_precision
from models import BaseModel
import argparse
from config_case import Config

from utils_case import log, get_L2reg

import warnings
warnings.filterwarnings("ignore")



# DAGFN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device, cuda:num or cpu")
    parser.add_argument('--hidden_dim1', type=int, default=512,help='Hidden units')
    parser.add_argument('--hidden_dim2', type=int, default=256,help='Hidden units')
    parser.add_argument('--proj_hidden_dim', type=int, default=256,help='project units')
    parser.add_argument('--n_layers', type=int, default=2,help='Number of hidden layers')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # torch.zeros((1)).to(device)
  
    config_file = "./config/case_study.ini"
    config = Config(config_file)

    fold_ACC = []
    fold_PRE = []
    fold_REC = []
    fold_F1 = []
    fold_SPEC = []
    fold_NEGPRE = []


    iter_acc, iter_pre, iter_rec, iter_f1, iter_spec, iter_negpre = [], [], [], [], [], []
    logfile = "log/new_log_seed_515_case_study.txt"
    for i in range(0, 1):
        fold_ACC = []
        fold_PRE = []
        fold_REC = []
        fold_F1 = []
        fold_SPEC = []
        fold_NEGPRE = []
        for fold in range(0,5):

            # case study: fastmode, no validation, epoch = 100
            config.structgraph_path = "case_study/alledg_139.txt"
            # train_path
            config.train_path = "case_study/miR139_id_train.txt"
            # test_path
            config.test_path = "case_study/miR139_id_test.txt"

            use_seed = not config.no_seed

            if use_seed:
                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
                torch.cuda.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)
                random.seed(config.seed)  
                torch.backends.cudnn.deterministic = True  
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
                os.environ['PYTHONHASHSEED'] = str(config.seed)
                torch.backends.cudnn.benchmark = False


            features, feature_demo, labels, idx_train, idx_test = load_data(config, device)
            tadj, fadj, ft_adj, ft_adj1, fadj1, tadj1, dropped_adj, added_adj, filtered_high_tensor, filtered_low_tensor  = ft_load_graph(config, features)

                            
            model = BaseModel(num_class=config.class_num,
                        fea_size=config.fdim,
                        hidden_dims = [args.hidden_dim1, args.hidden_dim2],
                        proj_dim = args.proj_hidden_dim,
                        num_layers = args.n_layers,
                        device = device,
                        dropout = config.dropout
                        )

            model = model.to(device)

            features = features
            tadj = tadj.to(device)
            fadj = fadj.to(device)
            ft_adj = ft_adj.to(device)

            dropped_adj = dropped_adj.to(device)
            added_adj = added_adj.to(device)


            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


            def train(model, epochs, in_adj, config):
              
                model.train()

          
                optimizer.zero_grad()

                # Train
                emb1 = model(features, in_adj) 
                emb2 = model(features, dropped_adj)
                emb3 = model(features, added_adj)

                # output = model.Mlp(emb1)

                k = torch.tensor(int(emb1.shape[0] * 1.0))
                p = (1/torch.sqrt(k))*torch.randn(k, emb1.shape[0]).to(device)

                emb2 = p @ emb2
                emb3 = p @ emb3
                emb2_aug2, emb3_aug3 = [model.project(x) for x in [emb2, emb3]]

                emb = torch.stack([emb1, emb2_aug2, emb3_aug3], dim=1)
                emb, att = model.Attention(emb)


                loss_drop = model.contrastive_loss(emb1, emb2_aug2, filtered_low_tensor)#, filtered_low_tensor
                loss_add = model.contrastive_loss(emb1, emb3_aug3, filtered_high_tensor)#, filtered_high_tensor
                loss_feature = model.contrastive_loss(emb2_aug2, emb3_aug3)

                contrast_loss = 1 * loss_drop + 1 * loss_add + 1 * loss_feature

                output = model.Mlp(emb)

                acc_train = accuracy(output[idx_train], labels[idx_train])
                loss_train1 = F.nll_loss(output[idx_train], labels[idx_train])
                L2_loss = 0.01*get_L2reg(model.parameters())
                loss_train = 1.0 * loss_train1 + config.lambd * contrast_loss + L2_loss
                
                
                loss_train.backward()  
                optimizer.step() 

                # Val
                # model.eval()
                # emb = model(features, in_adj) 
                # output = model.Mlp(emb)
                # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                # acc_val = accuracy(output[idx_val], labels[idx_val])

                if epochs == 0 and fold == 0:
            
                    log('case study, dropout_rate = 0.7, p_cut = 0.9, k = {:.1f}, p_rate = {:.1f}, l2_para = 0.01, contrast_loss = {:.4f}, temp = 0.6, lr = 0.0005:'
                        .format(config.k, config.p, config.lambd), logfile, sys.stdout, True)


         
                log('Epoch {:.1f}, loss_train: {:.4f}, acc_train: {:.4f}'
                        .format(epochs, loss_train.item(), acc_train.item()), logfile, sys.stdout, True)
                    
               
                

        
            def main_test(model, features, in_adj):
         
                model.eval()

                with torch.no_grad(): 
             
                    emb = model(features, in_adj) 
                    output = model.Mlp(emb)

                    predict = torch.argmax(output[idx_test], dim=1)
                    with open("case_miR139.txt", "w") as f:
                        for pred in predict:
                            f.write(f"{pred.item()}\n")

               
                    precision, recall, f1 = precision_recall_f1(output[idx_test], labels[idx_test])
                    specificity, neg_precision = calculate_specificity_and_negative_precision(output[idx_test], labels[idx_test])
                    acc_test = accuracy(output[idx_test], labels[idx_test])
                
                return precision, recall, f1, specificity, neg_precision, acc_test

            acc_max = 0
            pre = 0
            rec = 0
            f1s = 0
            spec = 0
            negpre = 0


            loss = []
            bad_counter = 0
            best = config.epochs + 1
            best_epoch = 0
     
            for epoch in range(config.epochs):
                in_adj = ft_adj
                train(model, epoch, in_adj, config)
               
            precision, recall, f1, specificity, neg_precision, acc_test = main_test(model, features, in_adj)
            # log("this is mRNA case study for BACH1, the precision is {:.4f}, recall is {:.4f}, F1 score is {:.4f}, specificity is {:.4f}, negative precision is {:.4f}, the accuray is {:.4f}".format(
            #                 fold, precision, recall, f1, specificity, neg_precision, acc_test), 
            #                 logfile, sys.stdout, True)
            
        







