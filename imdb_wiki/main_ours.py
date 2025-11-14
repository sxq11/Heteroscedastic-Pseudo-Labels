import argparse
import sys
import math
import os
import time
import random
import datetime
import tqdm

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import higher
from torch.utils.data import DataLoader, RandomSampler

import utils
from models import resnet50_unc, UncertaintyLearner
from datasets.imdb_wiki import get_imdbwiki
torch.multiprocessing.set_sharing_strategy("file_system")

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch for Heteroscedastic-Pseudo-Labels')
    
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help='data directory')
    parser.add_argument("--output_dir", type=str, default="./result",
                        help='directory to output the result')

    parser.add_argument("--pretrained", default=True, 
                        help='use pretrained model')
    parser.add_argument("--weights", default=None,
                        help='path to load specific weights')
    parser.add_argument("--run_test", default=True,
                        help='whether to run test after training')
    parser.add_argument("--test_only", default=False,
                        help='only run test without training')

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate for feature extractor")
    parser.add_argument('--fc_lr', type=float, default=1e-3,
                        help='learning rate for regression head')
    parser.add_argument('--unc_lr', type=float, default=1e-4,
                        help='learning rate for uncertainty-learner')
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_step_period", type=int, default=10,
                        help='epochs between each lr decay')
    parser.add_argument("--num_epochs", type=int, default=30,
                        help='number of total epochs to run') 
    parser.add_argument("--batch_size", type=int, default=48,
                        help='mini-batch size')

    parser.add_argument("--seed", type=int, default=0, help='manual seed')

    parser.add_argument('--gpu', default='4', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--num_workers", type=int, default=1,
                        help='dataloader threads')

    parser.add_argument("--labeled_ratio", type=float, default=0.1,
                        help='ratio of labeled data used in training')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='image size used in training')
    parser.add_argument("--y_mean", type=float, default=35.8602,
                        help='mean of training labels (for normalization)')
    parser.add_argument("--y_std", type=float, default=12.2532,
                        help='std of training labels (for normalization)') 
    parser.add_argument("--ssl_mult", type=int, default=2,
                        help='repeat labeled data by ssl_mult times')
    parser.add_argument("--w_ulb", type=float, default=10,
                        help='weight for unlabeled loss')
    parser.add_argument("--drp_p", type=float, default=0.05,
                        help='dropout rate to use in training')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print training loss every N iterations')

    args = parser.parse_args()
    return args

args = arg_parse()

if args.seed is None:
    args.seed = random.randint(1, 10000)
else:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()


def print_args(args):
    print("==========       CONFIG      =============")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))


def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    sys.stdout = utils.Logger(
        os.path.join(args.output_dir, 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))
    print_args(args)    
    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Data
    print('=====> Preparing data...')
    csv_path = os.path.join(args.data_dir, f"imdb_wiki.csv")
    dataset_train_lab, dataset_train_unlab, dataset_val, dataset_test = get_imdbwiki(
        csv_path, args.data_dir, args.img_size, labeled_ratio=args.labeled_ratio, ssl_mult=args.ssl_mult)

    dataloader_lb = DataLoader(
        dataset_train_lab, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    dataloader_unlb = DataLoader(
        dataset_train_unlab, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    generator = torch.Generator()
    generator.manual_seed(2025)
    dataloader_unc = DataLoader(dataset_train_lab, batch_size=args.batch_size, num_workers=args.num_workers, 
        sampler=RandomSampler(dataset_train_lab, generator=generator), shuffle=False, drop_last=True)    

    # Model
    model = resnet50_unc(pretrained=args.pretrained, drp_p = args.drp_p).cuda()
    unc_learner = UncertaintyLearner(input_dim=2, output_dim=1).cuda()

    # Load the pre_trained weights
    if args.weights is not None:
        checkpoint = torch.load(args.weights)
        if checkpoint.get('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            assert 1==2, "state dict not found"

    classifier_params = list(map(id, model.fc_m.parameters()))
    feat_params = filter(lambda p: id(p) not in classifier_params, model.parameters())

    optim_feat = torch.optim.Adam(feat_params, lr=args.lr, weight_decay=args.weight_decay)
    optim_fc = torch.optim.Adam(model.fc_m.parameters(), lr=args.fc_lr)
    optim_unc = torch.optim.Adam(unc_learner.parameters(), lr=args.unc_lr)

    if args.lr_step_period is None:
        args.lr_step_period = math.inf
    scheduler_feat = torch.optim.lr_scheduler.StepLR(optim_feat, args.lr_step_period)
    scheduler_fc = torch.optim.lr_scheduler.StepLR(optim_fc, args.lr_step_period)
    scheduler_unc = torch.optim.lr_scheduler.StepLR(optim_unc, args.lr_step_period)

    print("Run timestamp: {}\n".format(bkup_tmstmp))
    bestLoss = float("inf")

    if args.test_only:
        args.num_epochs = 0
    
    for epoch in range(0, args.num_epochs):
        print("Epoch #{}".format(epoch), flush=True) 

        start_time = time.time()
        if use_cuda:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)  
        
        # training phase          
        loss_tr, y_lb_pred, y_lb_true, y_ulb_true = run_epoch(model, unc_learner, dataloader_lb, dataloader_unlb, dataloader_unc, optim_feat, optim_fc, optim_unc)
        r2_value_0 = sklearn.metrics.r2_score(y_lb_true, y_lb_pred)
        print("Train: epoch_{}, loss_{:.4f}, r2_{:.3f}, time_{:.3f}\n".format(epoch, loss_tr, r2_value_0, time.time() - start_time))         
        
        scheduler_feat.step()
        scheduler_fc.step()
        scheduler_unc.step()

        # validation phase              
        loss_val, y_pred, y_true = run_epoch_val(model = model, dataloader = dataloader_val)
        r2_value = sklearn.metrics.r2_score(y_true, y_pred)
        print("Validate: epoch_{}, loss_{:.4f}, r2_{:.3f}, time_{:.4f}, size_{}\n".format(epoch, loss_val, r2_value, time.time() - start_time, y_true.size))

        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            "best_model_loss": loss_val,
            'r2': r2_value
        }
        # torch.save(save, os.path.join(args.output_dir, "checkpoint.pt"))
        
        if loss_val < bestLoss:
            print(f"R^2:{r2_value_0:.4f}, R^2_val:{r2_value:.4f}")
            print("saved best because {} < {}".format(loss_val, bestLoss))
            torch.save(save, os.path.join(args.output_dir, "best.pt"))
            bestLoss = loss_val
        
        total_loss, y_pred, y_true = run_epoch_val(model, dataloader_test)
        print("Test: {} (one clip) R2:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.r2_score(y_true, y_pred)))
        print("Test: {} (one clip) MAE:  {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.mean_absolute_error(y_true, y_pred)))
        print("Test: {} (one clip) RMSE: {:.4f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.mean_squared_error(y_true, y_pred)**0.5))
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best.pt"))
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    print("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]))
    if args.run_test:
        total_loss, y_pred, y_true = run_epoch_val(model, dataloader_test)
        print("Test: {} (one clip) R2:   {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.r2_score(y_true, y_pred)))
        print("Test: {} (one clip) MAE:  {:.4f}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.mean_absolute_error(y_true, y_pred)))
        print("Test: {} (one clip) RMSE: {:.4f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), sklearn.metrics.mean_squared_error(y_true, y_pred)**0.5))


def run_epoch(model, unc_learner, dataloader_lb, dataloader_unlb, dataloader_unc, optim_feat, optim_fc, optim_unc):
    total = 0  
    n = 0 

    y_lb_true, y_lb_pred = [], []
    y_ulb_true = []
    
    y_std = torch.tensor(args.y_std)
    y_mean = torch.tensor(args.y_mean)

    total_itr_num = len(dataloader_lb)
    dataloader_lb_itr = iter(dataloader_lb)
    dataloader_unlb_itr = iter(dataloader_unlb)
    dataloader_unc_itr = iter(dataloader_unc)

    model.train()
    unc_learner.train()
    torch.set_grad_enabled(True)
    for train_iter in range(total_itr_num):
        try:
            x_lb, y_lb = next(dataloader_lb_itr)
        except StopIteration:
            dataloader_lb_itr = iter(dataloader_lb)
            x_lb, y_lb = next(dataloader_lb_itr)

        x_lb, y_lb = x_lb.cuda(), y_lb.cuda().float()
        y_lb_true.append(y_lb.detach().cpu().numpy())
        y_lb_norm = ((y_lb - y_mean) / y_std).view(-1)

        try:
            (x_ulb_weak, x_ulb_strong), y_ulb = next(dataloader_unlb_itr)
        except StopIteration:
            dataloader_unlb_itr = iter(dataloader_unlb)
            (x_ulb_weak, x_ulb_strong), y_ulb = next(dataloader_unlb_itr)

        x_ulb_weak, x_ulb_strong = x_ulb_weak.cuda(), x_ulb_strong.cuda()
        y_ulb_true.append(y_ulb.detach().cpu().numpy())        

        try:
            x_unc, y_unc = next(dataloader_unc_itr)
        except StopIteration:
            dataloader_unc_itr = iter(dataloader_unc)
            x_unc, y_unc = next(dataloader_unc_itr)
        
        x_unc, y_unc = x_unc.cuda(), y_unc.cuda()
        y_unc_norm = ((y_unc - y_mean) / y_std).view(-1)

        optim_fc.zero_grad()
        if train_iter % 5 == 0:
            with higher.innerloop_ctx(model.fc_m, optim_fc) as (fmodel, diffort):
                _, feat_lb = model(x_lb) 
                preds_lb = fmodel(feat_lb.detach())
                del feat_lb
                loss_lb = F.mse_loss(preds_lb, y_lb_norm.unsqueeze(1))

                with torch.no_grad():
                    _, feats_ulb_weak = model(x_ulb_weak)
                    preds_ulb_weak = fmodel(feats_ulb_weak)  # pseudo label
                    del feats_ulb_weak
                _, feats_ulb_strong = model(x_ulb_strong)
                preds_ulb_strong = fmodel(feats_ulb_strong.detach())
                del feats_ulb_strong

                unc_input = preds_ulb_strong.detach() - preds_ulb_weak
                unc_input = torch.cat([unc_input, preds_ulb_strong.detach()], dim=-1)

                weight = unc_learner(unc_input.detach()) 
                unlabel_mse = (preds_ulb_strong - preds_ulb_weak) ** 2
                weight_new = (torch.exp(-weight)) / 2

                loss_ulb = torch.mean(weight_new * unlabel_mse)             
                loss = loss_lb + args.w_ulb * loss_ulb

                diffort.step(loss)

                _, feats_unc = model(x_unc)
                preds_unc = fmodel(feats_unc.detach())
                del feats_unc
                loss_unc = F.mse_loss(preds_unc, y_unc_norm.unsqueeze(1)) 

                optim_unc.zero_grad()
                loss_unc.backward()
                optim_unc.step()
        
        preds_lb, _ = model(x_lb)
        y_lb_pred.append(preds_lb.view(-1).to("cpu").detach() * y_std + y_mean)
        loss_lb = F.mse_loss(preds_lb, y_lb_norm.unsqueeze(1))

        with torch.no_grad():
            preds_ulb_weak, _ = model(x_ulb_weak)  # pseudo label
        preds_ulb_strong, _ = model(x_ulb_strong)
        
        with torch.no_grad():
            unc_input = (preds_ulb_strong - preds_ulb_weak).detach()
            unc_input = torch.cat([unc_input, preds_ulb_strong.detach()], dim=-1)
            weight = unc_learner(unc_input.detach())

        weight_new = torch.exp(-weight) / 2
        unlabel_mse = (preds_ulb_strong - preds_ulb_weak) ** 2

        loss_ulb = torch.mean(weight_new * unlabel_mse)
        loss = loss_lb + args.w_ulb * loss_ulb

        optim_fc.zero_grad()
        optim_feat.zero_grad()
        loss.backward()
        optim_fc.step()
        optim_feat.step()

        total += loss.item() * y_lb.size(0)
        n += y_lb.size(0)

        if train_iter % args.print_freq == 0:
            print("phase train itr {}/{}: ls {:.2f}({:.2f}) label_loss {:.4f} unlabeled_loss {:.4f}"
                  .format(train_iter, total_itr_num, 
                total / n, loss.item(), 
                loss_lb, loss_ulb, flush = True))
    
    y_lb_pred = np.concatenate(y_lb_pred)
    y_lb_true = np.concatenate(y_lb_true)
    y_ulb_true = np.concatenate(y_ulb_true)
    return total / n, y_lb_pred, y_lb_true, y_ulb_true
    
    
def run_epoch_val(model, dataloader):
    total = 0 
    n = 0   

    y_true, y_pred = [], []
    y_std = torch.tensor(args.y_std)
    y_mean = torch.tensor(args.y_mean)
    
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (inputs, targets) in dataloader:

                y_true.append(targets.numpy())
                inputs = inputs.cuda()
                targets = targets.cuda()

                preds, _ = model(inputs)
                
                targets = ((targets - y_mean) / y_std).unsqueeze(1)
                y_pred.append(preds.view(-1).to("cpu").detach() * y_std + y_mean)
                
                loss = F.mse_loss(preds, targets) 

                total += loss.item() * inputs.size(0)
                n += inputs.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))
                pbar.update()

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    return total / n, y_pred, y_true


if __name__ == '__main__':
    main()