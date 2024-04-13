from __future__ import print_function

import os
import sys
import argparse
import time
import math

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.utils.data as data
from torch.distributions import Categorical

import datasets.froth_data as dataset

from util import AverageMeter
from util import adjust_learning_rate, accuracy
from util import save_model

from models import hf_model
from InceptionV3_Head import InceptionV3_Head
from wideresnet import WideResNet
from prob_memory_network import Memory_Network

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=40, help='batch_size')
    parser.add_argument('--n_labeled', type=int, default=6000, help='num of labeled data')
    parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--train-iteration', type=int, default=500, help='Number of iteration per epoch')

    # memory
    parser.add_argument('--mem_size', type=int, default=1000, help='number of memory slots') #2000
    parser.add_argument('--key_dim', type=int, default=128, help='key dimension')
    parser.add_argument('--val_dim', type=int, default=6, help='dimension of class distribution')
    parser.add_argument('--top_k', type=int, default=128, help='top_k for memory reading') #200
    parser.add_argument('--lambd', type=float, default=0.9, help='momemtun of memory update') #260
    parser.add_argument('--val_thres', type=float, default=0.07, help='threshold for value matching') #0.09-->85%(450epo)
    # 0.15-->49%    0.09-->60%    0.07-->60%
    parser.add_argument('--age_noise', type=float, default=8.0, help='number of training epochs')
    parser.add_argument('--tau', type=float, default=0.005, help='step for momentum update')
    
    parser.add_argument('--temp', type=float, default=1.0, help='temperature for more concentration')
    parser.add_argument('--T', default=0.5, type=float, help='temperature for sharping')
    
    parser.add_argument('--alpha', type=float, default=0.75, help='weight for mixup')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    parser.add_argument('--cosine', default=True, action='store_true', help='using cosine annealing')

    # model dataset
    parser.add_argument('--dataset', type=str, default='froth', help='dataset')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/{}_models'.format(opt.dataset)
    opt.tb_path = './save/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'lr_{}_bsz_{}'.\
        format(opt.learning_rate, opt.batch_size)
    
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    train_labeled_set, train_unlabeled_set, test_set = dataset.get_froth_data(opt.n_labeled)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=2*opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=176, shuffle=False, num_workers=4)

    return labeled_trainloader, unlabeled_trainloader, test_loader


def set_model(opt):
    model = Memory_Network(InceptionV3_Head, mem_size=opt.mem_size, key_dim=opt.key_dim, val_dim=opt.val_dim, top_k=opt.top_k, lambd=opt.lambd, age_noise=opt.age_noise)
    
    Prob_Pred = hf_model()
    Prob_Pred.load_state_dict(torch.load('./save/trained_hf_model_epoch_300.pth'))
    
    if torch.cuda.is_available():
        model = model.cuda()
        Prob_Pred = Prob_Pred.cuda()
        cudnn.benchmark = True
        
    Prob_Pred.eval()
    
    return model, Prob_Pred


def resume(labeled_trainloader, unlabeled_trainloader, model, Prob_Pred, opt):
    """one epoch training"""
    model.eval()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    for batch_idx in range(opt.train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        #targets_x = torch.zeros(inputs_x.size(0), 10).scatter_(1, targets_x.view(-1,1).long(), 1)

        if torch.cuda.is_available():
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            query_u, key_u = model(inputs_u)
            outputs_u = model.predict(query_u)
            query_u2, key_u2 = model(inputs_u2)
            outputs_u2 = model.predict(query_u2)
            p = (outputs_u + outputs_u2) / 2
            pt = p**(1/opt.T)
            prob_u = pt / pt.sum(dim=1, keepdim=True)
            prob_u = prob_u.detach()
            
            query_x, key_x = model(inputs_x)
            prob_x = F.softmax(Prob_Pred(inputs_x), dim=1)
            
            query = torch.cat((query_u, query_u2, query_x), dim=0)
            key = torch.cat((key_u, key_u2, key_x), dim=0)
            prob = torch.cat((prob_u, prob_u, prob_x), dim=0)
            model.memory_update(query, key, prob, opt.val_thres)
            


def train(labeled_trainloader, unlabeled_trainloader, model, Prob_Pred, ce_criterion, optimizer, epoch, opt, tb):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    sup_ce_losses = AverageMeter()
    con_sys_losses = AverageMeter()
    pred_ent_losses = AverageMeter()
    contrast_losses = AverageMeter()
    top1 = AverageMeter()
    top1_aug = AverageMeter()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    end = time.time()
    
    for batch_idx in range(opt.train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), targets_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), targets_u = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        #targets_x = torch.zeros(inputs_x.size(0), 10).scatter_(1, targets_x.view(-1,1).long(), 1)

        if torch.cuda.is_available():
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            targets_u = targets_u.cuda()

        with torch.no_grad():
            prob_x = 0.5*F.softmax(Prob_Pred(inputs_x), dim=1) + 0.5*targets_x
            # compute guessed labels of unlabel samples
            query_u, _ = model(inputs_u)
            outputs_u = model.predict(query_u)
            query_u2, _ = model(inputs_u2)
            outputs_u2 = model.predict(query_u2)
            p = (outputs_u + outputs_u2) / 2
            pt = p**(1/opt.T)
            prob_u = pt / pt.sum(dim=1, keepdim=True)
            prob_u = prob_u.detach()
            
        # mixup - optional
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_prob = torch.cat([prob_x, prob_u, prob_u], dim=0)

        l = np.random.beta(opt.alpha, opt.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        prob_a, prob_b = all_prob, all_prob[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_prob = l * prob_a + (1 - l) * prob_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        mixed_query, mixed_key = model(mixed_input[0])
        logits = [model.predict(mixed_query)]
        for inputs in mixed_input[1:]:
            mixed_query, mixed_key = model(inputs)
            logits.append(model.predict(mixed_query))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        
        ##########      COMPUTE SEMI-SUPERVISED LOSSES      ##########
        # 1) compute supervised cross-entropy loss
        query_x, key_x = model(inputs_x)
        pred_x = model.predict(query_x)
        sup_ce_loss_1 = ce_criterion(pred_x, targets_x)
        sup_ce_loss_2 = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_prob[:batch_size], dim=1))
        sup_ce_loss = sup_ce_loss_1 + sup_ce_loss_2
        
        # 2) consistency regularization
        query_u, key_u = model(inputs_u)
        pred_u = model.predict(query_u)
        '''with torch.no_grad():
            query_u2, key_u = model(inputs_u2)
            pred_u2 = model.predict(query_u2)
        con_sys_loss = torch.mean((pred_u - pred_u2)**2)'''
        sup_unlabeled = ce_criterion(pred_u, targets_u)
        con_sys_loss = torch.mean((logits_u - mixed_prob[batch_size:])**2)
        
        # 3) mutual information
        #pred_ent_loss = torch.mean(Categorical(probs=logits_u).entropy())
        pred_ent_loss = (-logits_u * torch.log(logits_u + 1e-5)).sum(1).mean()
        
        # 4) contrastive learning
        contrast_loss = model.contrast_loss(query_u, query_u2)
    
        loss = sup_ce_loss + sup_unlabeled + linear_rampup(epoch) * (0.5*con_sys_loss + 0.1*pred_ent_loss + 0.15*contrast_loss) #0.15-->0.4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ##########      UPDATE MOMERIES      ##########
        with torch.no_grad():
            model.key_enc_update()
            query_x, key_x = model(inputs_x)
            model.memory_update(query_x, key_x, prob_x, opt.val_thres)
            
            query_u, key_u = model(inputs_u)
            model.memory_update(query_u, key_u, prob_u, opt.val_thres)

        # update metric
        acc1, acc5 = accuracy(pred_x, targets_x, topk=(1, 5))
        top1.update(acc1[0], pred_x.size(0))
        
        acc_aug, _ = accuracy(prob_x, targets_x, topk=(1, 5))
        top1_aug.update(acc_aug[0], pred_x.size(0))
        
        sup_ce_losses.update(sup_ce_loss.item(), batch_size)
        con_sys_losses.update(con_sys_loss.item(), batch_size)
        pred_ent_losses.update(pred_ent_loss.item(), batch_size)
        contrast_losses.update(contrast_loss.item(), batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        tb.add_scalar("sup_ce_losses", sup_ce_losses.avg, epoch)
        tb.add_scalar("con_sys_losses", con_sys_losses.avg, epoch)
        tb.add_scalar("pred_ent_loss", pred_ent_losses.avg, epoch)
        tb.add_scalar("contrast_losses", contrast_losses.avg, epoch)
        tb.add_scalar("top1", top1.avg, epoch)
        if (batch_idx + 1) % opt.print_freq == 0:
            #print('Hello {Jin}/t'.format(Jin='Jin'))中/t是指sep='/t',即一个Tab的间距
            print('Train: [{0}][{1}/{2}]   '
                  'BT {batch_time.avg:.3f}   '
                  'CE {ce.avg:.3f}   '
                  'CS {cs.avg:.4f}   '
                  'ET {et.avg:.3f}   '
                  'CT {ct.avg:.3f}   '
                  'Acc {top1.avg:.3f}'.format(
                   epoch, batch_idx + 1, opt.train_iteration, batch_time=batch_time, ce=sup_ce_losses, cs=con_sys_losses, et=top1_aug, ct=contrast_losses, top1=top1)) #et=pred_ent_losses
            sys.stdout.flush()

    return sup_ce_losses.avg, top1.avg


def validate(valloader, model, Prob_Pred, criterion, epoch, opt, tb):
    """one epoch training"""
    model.eval()
    
    y_pred = []
    y_true = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            query, key = model(inputs)
            pred = model.predict(query)
            
            loss = criterion(pred, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(pred, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
            labels = targets.data.cpu().numpy()
            y_pred.extend(output)
            y_true.extend(labels)
            

            # print info
            tb.add_scalar("test_losses", losses.avg, epoch)
            tb.add_scalar("test_top1", top1.avg, epoch)
            if (batch_idx + 1) % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   batch_idx + 1, len(valloader), loss=losses, top1=top1))
                sys.stdout.flush()
                
    classes = ('Class I', 'Class II', 'Class III', 'Class IV', 'Class V', 'Class VI')
    cf_matrix = confusion_matrix(y_true, y_pred)            
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,10))
    sn.heatmap(df_cm, linewidth=.5, annot=True, cmap="GnBu", annot_kws={"fontsize":16})
    plt.savefig(f"output-{epoch}.png", dpi=300)

    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="Froth")

    # build data loader
    labeled_trainloader, unlabeled_trainloader, test_loader = set_loader(opt)
    opt.train_iteration = len(unlabeled_trainloader)+1

    # build model and criterion
    model, Prob_Pred = set_model(opt)
    
    resume(labeled_trainloader, unlabeled_trainloader, model, Prob_Pred, opt)

    # build optimizer
    optimizer = optim.Adam(model.encoder_q.parameters(), lr = opt.learning_rate)
    #base_optimizer = torch.optim.SGD
    #optimizer = SAM(model.encoder_q.parameters(), base_optimizer, rho=0.5, adaptive=True, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    ce_criterion = torch.nn.CrossEntropyLoss()
    ce_criterion = ce_criterion.cuda()
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(labeled_trainloader, unlabeled_trainloader, model, Prob_Pred, ce_criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        loss, val_acc = validate(test_loader, model, Prob_Pred, ce_criterion, epoch, opt, tb)
        
        #if val_acc > best_acc:
        #    best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'query_enc_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.encoder_q.state_dict(), save_file)
            save_file = os.path.join(
                opt.save_folder, 'key_enc_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.encoder_k.state_dict(), save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))

    
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length=40):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
                     
                     
if __name__ == '__main__':
    main()
