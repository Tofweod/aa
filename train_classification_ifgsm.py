"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import pdb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import seaborn as sns
import wandb
import socket

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from attackers.attacker_utils import get_label_dict, save_tensor2txt, save_tensor2obj

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'attackers'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--no_save_result',action='store_true', default=False, help='dont save results')
    parser.add_argument('--save_fmt', default='obj', type=str, help='perturb data save format')
    ''' ATTACKER SETTINGS '''
    parser.add_argument('--attacker', default='basic_ifgsm', help='attacker name [default: basic_ifgsm]')
    parser.add_argument('--eps', default=10.0, type=float,help='eps for iteration termination')
    parser.add_argument('--nb_iter', default=10, type=int,help='number of iteration in ifgsm')
    parser.add_argument('--alpha', default=0.01,type=float,help='alpha for iterative perturb')
    # TODO:
    parser.add_argument('--fixed', default=0.3,type=float,help='fixed proporation')
    #  parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    #  parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    #  parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def train_attack(save_fn,attack,loader,label_dict,num_class=40):

    # TODO: train or eval
    #  pdb.set_trace()
    attack.predict.eval()
    global_idx = 1

    attack_dist = []
    attack_success = 0
    

    pred_matrix = np.zeros((num_class,num_class))
    pred_num = np.zeros(num_class)

    for _,(points,target) in tqdm(enumerate(loader),total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
    
        # TODO: draw loss and heatmap of difference
        perturb_points,pred,dist = attack(points,target,num_class=num_class)
        perturb_target = pred.data.max(1)[1]

        attack_dist.append(torch.mean(dist).cpu().item())
        attack_success += torch.sum(torch.ne(perturb_target,target)).cpu().item()


        perturb_points = perturb_points.transpose(2,1).cpu()
        perturb_target = perturb_target.cpu()
        target = target.cpu()

        dist =  dist.cpu()

        pred = pred.detach().cpu()


        pred = pred.numpy()

        if args.save_fmt == 'obj':
            save_func = save_tensor2obj
        else:
            save_func = save_tensor2txt

        bsz = points.size()[0]
        for i in range(bsz):
            org = target[i].item()
            pred_matrix[org] += pred[i]
            pred_num[org] += 1
            if not args.no_save_result:
                save_func(save_fn,
                        perturb_points[i],
                        label_dict[perturb_target[i].item()],
                            label_dict[target[i].item()],
                        global_idx)
            global_idx += 1


    # draw heatmap
    mean_matrix = np.zeros_like(pred_matrix)
    mean_matrix = pred_matrix / pred_num[:,np.newaxis]
    #  for i in range(num_class):
        #  mean_matrix[i] = pred_matrix[i] / pred_num[i]

    if args.num_category == 40:
        plt.figure(figsize=(24,18))

    hm = sns.heatmap(mean_matrix, cmap='YlGnBu', annot=True)
    hm.set_xticklabels(list(label_dict.values()),rotation=-45,fontsize=8)
    hm.set_yticklabels(list(label_dict.values()),rotation=-45,fontsize=8)

    mean_attack_dist = sum(attack_dist)/ len(attack_dist)
    mean_attack_success = attack_success / global_idx
    wandb.log({"mean_attack_success":mean_attack_success})
    wandb.log({"mean_attack_dist":mean_attack_dist})
    wandb.log({"diff_matrix":wandb.Image(hm)})
    


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    """WANDB INIT"""
    wandb_run_name = "{}_{}_modelnet{}_b{}_eps{}_it{}_a{}".format(args.attacker,args.model,args.num_category,
                                                              args.batch_size,args.eps,args.nb_iter,args.alpha)

    if args.use_normals:
        wandb_run_name += "_use_normals"
    if args.use_uniform_sample:
        wandb_run_name += "_uniform_sample"

    wandb_run_name += "_{}".format(timestr)
    wandb.init(project="aa",name=wandb_run_name,config=args,notes=socket.gethostname(),job_type="training")



    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    '''ATTACKER LOADING'''
    attacker = importlib.import_module(args.attacker)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)


    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

   
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    attack = attacker.get_attacker(classifier,criterion,args.eps,args.nb_iter,args.alpha,args.fixed)
    
    log_string("Test start")
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    log_string("Test finish")

    # TODO:
    save_fn = str(experiment_dir) + '/results'
    os.makedirs(save_fn,exist_ok=True)
    idx_name_dict = get_label_dict(test_dataset.classes)
    train_attack(save_fn,attack,testDataLoader,idx_name_dict,num_class=num_class)

    wandb.finish()
    #  '''TRANING'''

    #  try:
        #  checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        #  start_epoch = checkpoint['epoch']
        #  classifier.load_state_dict(checkpoint['model_state_dict'])
        #  log_string('Use pretrain model')
    #  except:
        #  log_string('No existing model, starting training from scratch...')
        #  start_epoch = 0
#  

    # no need for optimizer
    
    #  if args.optimizer == 'Adam':
        #  optimizer = torch.optim.Adam(
            #  classifier.parameters(),
            #  lr=args.learning_rate,
            #  betas=(0.9, 0.999),
            #  eps=1e-08,
            #  weight_decay=args.decay_rate
        #  )
    #  else:
        #  optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    #  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    #  global_epoch = 0
    #  global_step = 0
    #  best_instance_acc = 0.0
    #  best_class_acc = 0.0
#  
    #  '''TRANING'''
    #  logger.info('Start training...')
    #  for epoch in range(start_epoch, args.epoch):
        #  log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        #  mean_correct = []
        #  classifier = classifier.train()
#  
        #  #  scheduler.step()
        #  for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            #  #  optimizer.zero_grad()
#  
            #  points = points.data.numpy()
            #  points = provider.random_point_dropout(points)
            #  points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            #  points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            #  points = torch.Tensor(points)
            #  points = points.transpose(2, 1)
#  
            #  if not args.use_cpu:
                #  points, target = points.cuda(), target.cuda()
#  
#  
            #  perturb_data = attack(points,target)
#  
            #  with torch.no_grad():
                #  pred, _ = classifier(perturb_data)
                #  pred_choice = pred.data.max(1)[1]
#  

            #  pred, trans_feat = classifier(points)
            #  loss = criterion(pred, target.long(), trans_feat)
            #  pred_choice = pred.data.max(1)[1]
#  
            #  correct = pred_choice.eq(target.long().data).cpu().sum()
            #  mean_correct.append(correct.item() / float(points.size()[0]))
            #  loss.backward()
            #  optimizer.step()
            #  global_step += 1
#  
        #  train_instance_acc = np.mean(mean_correct)
        #  log_string('Train Instance Accuracy: %f' % train_instance_acc)
#  
        #  with torch.no_grad():
            #  instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
#  
            #  if (instance_acc >= best_instance_acc):
                #  best_instance_acc = instance_acc
                #  best_epoch = epoch + 1
#  
            #  if (class_acc >= best_class_acc):
                #  best_class_acc = class_acc
            #  log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            #  log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
#  
            #  if (instance_acc >= best_instance_acc):
                #  logger.info('Save model...')
                #  savepath = str(checkpoints_dir) + '/best_model.pth'
                #  log_string('Saving at %s' % savepath)
                #  state = {
                    #  'epoch': best_epoch,
                    #  'instance_acc': instance_acc,
                    #  'class_acc': class_acc,
                    #  'model_state_dict': classifier.state_dict(),
                    #  'optimizer_state_dict': optimizer.state_dict(),
                #  }
                #  torch.save(state, savepath)
            #  global_epoch += 1
#  
    #  logger.info('End of training...')
#  

if __name__ == '__main__':
    args = parse_args()
    main(args)
