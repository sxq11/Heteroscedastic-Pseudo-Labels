import os
import sys
import time
import random
import tqdm
import math
import argparse
import shutil
import codecs
import logging
import torch
import datetime
import higher
import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.jit as jit
from allennlp.data.data_loaders import SimpleDataLoader

from datasets.get_sts import build_tasks
from model.models import build_model
from utils.util import Logger
import torch.backends.cudnn as cudnn
from scipy.stats import pearsonr, spearmanr
from model.uncertainty_learner import UncertaintyLearner


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch for Heteroscedastic-Pseudo-Labels')
    
    parser.add_argument("--data_dir", type=str, default="./glue_data/STS-B",
                        help='path to dataset')
    parser.add_argument("--output_dir", type=str, default="./result", 
                        help='Directory to output the result')
     
    parser.add_argument('--max_seq_len', type=int, default=40,
                        help='max sequence length')  
    parser.add_argument('--max_word_v_size', type=int, default=30000,
                        help='max word vocab size') 
    
    parser.add_argument('--dropout_embs', type=float, default=0.2,
                        help='dropout rate for embeddings')
    parser.add_argument('--d_word', type=int, default=300,
                        help='dimension of word embeddings')
    parser.add_argument('--glove', type=int, default=1,
                        help='1 if use glove, else from scratch')
    parser.add_argument('--train_words', type=int, default=0,
                        help='1 if make word embs trainable')
    parser.add_argument('--word_embs_file', type=str, default='./glove/glove.840B.300d.txt', 
                        help='file containing word embs')
    
    parser.add_argument('--d_hid', type=int, default=1024,
                        help='hidden dimension size',)
    parser.add_argument('--n_layers_enc', type=int, default=2,
                        help='number of RNN layers')
    parser.add_argument('--n_layers_highway', type=int, default=0,
                        help='number of highway layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate to use in training')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="learning rate for feature extractor")
    parser.add_argument('--fc_lr',type=float, default=1e-3,
                        help='learning rate for regression head')
    parser.add_argument('--unc_lr', type=float, default=1e-4,
                        help='learning rate for uncertainty-learner')
    parser.add_argument("--num_epochs", type=int, default=200,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    
    parser.add_argument("--seed", type=int, default=0, help='manual seed')
    
    parser.add_argument('--gpu', type=int, default=2, 
                        help='-1 if no CUDA, else gpu id (single gpu is enough)')
    
    parser.add_argument('--labeled_ratio', type=float, default=0.1,
                        help='ratio of labeled data used in training')
    parser.add_argument("--w_ulb", type=float, default=1.0,
                        help='weight for unlabeled loss')
    parser.add_argument("--y_max", type=float, default=5.0,
                        help="maximum label value (used as scaling factor in label normalization")
    parser.add_argument('--lambda2', default=1.0, type=float,
                        help='weight for the regularization term')  
    parser.add_argument('--print_freq',type=int, default=50,
                        help='print training loss every N iterations')
    args = parser.parse_args()
    return args

args = arg_parse()

os.environ['CUDA_VISIBLE_DEVICES'] = 'args.gpu'
use_cuda = torch.cuda.is_available()
if args.gpu != -1 and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')

if args.seed is None:
    args.seed = random.randint(1, 10000)
else:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def print_args(args):
    print("==========       CONFIG      =============")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))

def make_loader(data, batch_size, shuffle, device):
    loader = SimpleDataLoader(data, batch_size=batch_size, shuffle=shuffle)
    loader.set_target_device(device)
    return loader

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    sys.stdout = Logger(os.path.join(args.output_dir,  'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))   
    print_args(args)

    # Data preprocessing
    csv_path = os.path.join(args.data_dir, f"sts.tsv")    
    tasks, vocab, word_embs = build_tasks(args, csv_path, max_seq_len= args.max_seq_len, labeled_ratio=args.labeled_ratio)
    
    # Model 
    model = build_model(args, vocab, word_embs).to(device)
    unc_learner = UncertaintyLearner(input_dim=2, output_dim=1).to(device)
    
    classifier_params = list(map(id, model.pred_layer.parameters()))
    feat_params = filter(lambda p: id(p) not in classifier_params, model.parameters())

    optim_feat = torch.optim.Adam(feat_params, lr=args.lr, weight_decay=1e-5)
    optim_unc = torch.optim.Adam(unc_learner.parameters(), lr=args.unc_lr, weight_decay=1e-5)
    optim_fc = torch.optim.Adam(model.pred_layer.parameters(), lr=args.fc_lr, weight_decay=1e-5)
    
    # Data 
    dataloader_lb = make_loader(tasks.label_data, args.batch_size, True, device)
    dataloader_unlb = make_loader(tasks.unlabel_data, args.batch_size, True, device)
    dataloader_val = make_loader(tasks.val_data, args.batch_size, False, device)
    dataloader_test = make_loader(tasks.test_data, args.batch_size, False, device)

    torch.manual_seed(args.seed + 2024)
    dataloader_unc = make_loader(tasks.label_data, args.batch_size, True, device) 

    best_pearson = float("inf")
    for epoch in range(0, args.num_epochs): 
        print("Epoch #{}".format(epoch), flush=True)
        start_time = time.time()
        
        # training phase
        loss, y_lb_true, y_lb_pred, y_ulb_true = run_epoch(model, unc_learner, dataloader_lb, dataloader_unlb, dataloader_unc, optim_feat, optim_unc, optim_fc)   
        pearson, _ = pearsonr(y_lb_pred, y_lb_true)
        print("Train: epoch_{}, loss_{:.4f}, pearson_{:.4f}, time_{:.3f}\n".format(epoch, loss, pearson, time.time() - start_time))

        # validation phase
        loss_val, y_true, y_pred = run_epoch_val(model, dataloader_val)
        pearson_val, _ = pearsonr(y_pred, y_true)
        min_pearson = 1 - pearson_val
        print("Validate: epoch_{}, loss_{:.4f}, pearson_val_{:.4f}, time_{:.4f}".format(epoch, loss_val, pearson_val, time.time() - start_time))
        
        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_model_loss': loss_val,
            'pearson': pearson_val
        }
        # torch.save(save, os.path.join(args.output_dir, "checkpoint.pt"))
        
        if min_pearson < best_pearson:
            print(f"pearson:{pearson:.4f}, pearson_val:{pearson_val:.4f}")
            print("saved best because {} > {}".format(pearson_val, 1-best_pearson))
            torch.save(save, os.path.join(args.output_dir, "best.pt"))      
            best_pearson = min_pearson     
                
        total_loss, y_true, y_pred  = run_epoch_val(model, dataloader_test)
        pearson, _ = pearsonr(y_pred, y_true)
        Spearman, _ = spearmanr(y_pred, y_true)
        MAE = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        MSE = sklearn.metrics.mean_squared_error(y_true, y_pred)
        R2 = sklearn.metrics.r2_score(y_true, y_pred)
        print("Test: {} (one clip) R2:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), R2))
        print("Test: {} (one clip) pearson:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), pearson))
        print("Test: {} (one clip) spearman:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), Spearman))
        print("Test: {} (one clip) MAE:  {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), MAE))
        print("Test: {} (one clip) MSE: {:.4f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), MSE))
         
    checkpoint = torch.load(os.path.join(args.output_dir, "best.pt"))
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    print("Best validation loss {} from epoch {}, pearson {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["pearson"]))
    total_loss, y_true, y_pred  = run_epoch_val(model, dataloader_test)
    pearson, _ = pearsonr(y_pred, y_true)
    Spearman, _ = spearmanr(y_pred, y_true)
    MAE = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    MSE = sklearn.metrics.mean_squared_error(y_true, y_pred)
    R2 = sklearn.metrics.r2_score(y_true, y_pred)
    print("Test: {} (one clip) R2:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), R2))
    print("Test: {} (one clip) pearson:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), pearson))
    print("Test: {} (one clip) spearman:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), Spearman))
    print("Test: {} (one clip) MAE:  {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), MAE))
    print("Test: {} (one clip) MSE: {:.4f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), MSE))
 
              
def run_epoch(model, unc_learner, dataloader_lb, dataloader_unlb, dataloader_unc, optim_feat, optim_unc, optim_fc):
    y_lb_true, y_lb_pred = [], []
    y_ulb_true = []
    y_max = torch.tensor(args.y_max).to(device)
    
    total_iter_num = len(dataloader_unlb)
    dataloader_lb_iter = iter(dataloader_lb)
    dataloader_unlb_iter = iter(dataloader_unlb)
    dataloader_unc_iter = iter(dataloader_unc)

    model.train()  
    unc_learner.train()   
    for train_iter in range(total_iter_num):
        try:
            instance_lb = next(dataloader_lb_iter)
        except:
            dataloader_lb_iter = iter(dataloader_lb)
            instance_lb = next(dataloader_lb_iter)
        x_lb1 = instance_lb['input1']
        x_lb2 = instance_lb['input2']
        y_lb = instance_lb['label']
        y_lb_true.append(y_lb.view(-1).cpu().detach().numpy())
        y_lb = y_lb / y_max
        
        try:
            instance_ulb = next(dataloader_unlb_iter)
        except:
            dataloader_unlb_iter = iter(dataloader_unlb)
            instance_ulb= next(dataloader_unlb_iter)
        x_ulb_weak1 = instance_ulb['input1_weak']
        x_ulb_weak2 = instance_ulb['input2_weak']
        x_ulb_strong1 = instance_ulb['input1_strong']
        x_ulb_strong2 = instance_ulb['input2_strong']       
        y_ulb = instance_ulb['label']
        y_ulb_true.append(y_ulb.view(-1).cpu().detach().numpy())
        y_ulb = y_ulb / y_max
        
        try:
            instance_unc = next(dataloader_unc_iter)
        except:
            dataloader_unc_iter = iter(dataloader_unc)
            instance_unc = next(dataloader_unc_iter)
        x_unc1 = instance_unc['input1']
        x_unc2 = instance_unc['input2']
        y_unc = instance_unc['label']
        y_unc = y_unc / y_max
        
        optim_fc.zero_grad()  
        if train_iter % 5 == 0:
            with higher.innerloop_ctx(model.pred_layer, optim_fc) as (fmodel, diffort):
                out_dict_label = model(input1=x_lb1, input2=x_lb2, label=y_lb, return_feat=True)
                feats_lb = out_dict_label['feat']
                preds_lb = fmodel(feats_lb)
                loss_lb = F.mse_loss(preds_lb, y_lb)
                
                with torch.no_grad():
                    out_dict_weak = model(input1=x_ulb_weak1, input2=x_ulb_weak2, label=y_ulb, return_feat=True)    
                    feats_ulb_weak = out_dict_weak['feat']
                    preds_ulb_weak = fmodel(feats_ulb_weak) 
                out_dict_strong = model(input1=x_ulb_strong1, input2=x_ulb_strong2, label=y_ulb, return_feat=True)
                feats_ulb_strong = out_dict_strong['feat']
                preds_ulb_strong = fmodel(feats_ulb_strong)  

                unc_input = (preds_ulb_strong - preds_ulb_weak).detach()
                unc_input = torch.cat([unc_input, preds_ulb_strong.detach()], dim=-1)
                weight = unc_learner(unc_input.detach())
                unlabel_mse = (preds_ulb_strong - preds_ulb_weak) ** 2
                weight_new = torch.exp(-weight) / 2

                loss_ulb = torch.mean(weight_new * unlabel_mse)            
                loss = loss_lb + args.w_ulb * loss_ulb
                
                diffort.step(loss)
                
                out_unc = model(input1=x_unc1, input2=x_unc2, label=y_unc, return_feat=True)
                feats_unc = out_unc['feat']
                preds_unc = fmodel(feats_unc.detach())
                unc_input = (preds_unc - y_unc).detach()
                unc_input = torch.cat([unc_input, preds_unc.detach()], dim=-1)
                weight = unc_learner(unc_input.detach()) 
                loss_unc = F.mse_loss(preds_unc, y_unc) -  args.lambda2*torch.mean(weight)
                
                optim_unc.zero_grad()
                loss_unc.backward()
                optim_unc.step()
                       
        out_dict_label = model(input1=x_lb1, input2=x_lb2, label=y_lb)
        preds_lb = out_dict_label['logits']
        loss_lb = F.mse_loss(preds_lb, y_lb)
        preds_lb = out_dict_label['logits'] * y_max
        y_lb_pred.append(preds_lb.view(-1).cpu().detach().numpy())
        
        with torch.no_grad():
            out_dict_weak = model(input1=x_ulb_weak1, input2=x_ulb_weak2, label=y_ulb)    
            preds_ulb_weak = out_dict_weak['logits'] 
        out_dict_strong = model(input1=x_ulb_strong1, input2=x_ulb_strong2, label=y_ulb)
        preds_ulb_strong = out_dict_strong['logits']  

        with torch.no_grad():
            unc_input = (preds_ulb_strong - preds_ulb_weak).detach()
            unc_input = torch.cat([unc_input, preds_ulb_strong.detach()], dim=-1)
            weight = unc_learner(unc_input.detach())  

        unlabel_mse = (preds_ulb_strong - preds_ulb_weak) ** 2
        weight_new = torch.exp(-weight) / 2
        
        loss_ulb = torch.mean(weight_new * unlabel_mse) 

        loss = loss_lb + args.w_ulb * loss_ulb 
       
        optim_fc.zero_grad()
        optim_feat.zero_grad()
        loss.backward()
        optim_fc.step()
        optim_feat.step()
        
        if train_iter % args.print_freq == 0:
            print("train itr {}/{} : loss {:.4f} label_loss {:.4f} unlabel_loss {:.4f} ".format(train_iter, total_iter_num, loss.item(), loss_lb, loss_ulb, flush=True))   
        
    y_lb_true = np.concatenate(y_lb_true)
    y_lb_pred = np.concatenate(y_lb_pred)
    y_ulb_true = np.concatenate(y_ulb_true)
    return loss, y_lb_true, y_lb_pred, y_ulb_true


def run_epoch_val(model, dataloader):
    y_true, y_pred = [], []
    y_max = torch.tensor(args.y_max).to(device)
    dataloader_itr = iter(dataloader)
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i in range(len(dataloader)):
                try:
                    instance = next(dataloader_itr)
                except:
                    dataloader_itr = iter(dataloader)
                    instance = next(dataloader_itr)
                input1 = instance['input1']
                input2 = instance['input2']
                label = instance['label'] 
                y_true.append(label.view(-1).cpu().numpy())
                label = instance['label'] / y_max
                
                out_dict = model(input1=input1, input2=input2, label=label) 
                preds = out_dict['logits']
                loss = F.mse_loss(preds, label)
                y_pred.append((preds * y_max).view(-1).cpu().numpy())
                pbar.set_postfix_str("{:.2f}".format(loss.item()))
                pbar.update()
                
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
     
    return loss.item(), y_true, y_pred
     
     
if __name__ == '__main__':
    main()
