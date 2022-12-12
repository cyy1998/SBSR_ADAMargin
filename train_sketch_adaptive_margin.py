# -*- coding: utf-8 -*-
import os
import argparse
from random import sample,randint
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.sketch_model import SketchModel
from model.view_model import MVCNN
from model.magclassifier import Classifier
from model.uncer_classifer import L2Classifier
from loss.am_softmax import AMSoftMaxLoss
from loss.magface import MagFaceLoss

parser = argparse.ArgumentParser("Sketch_View Modality")
# dataset
parser.add_argument('--sketch-datadir', type=str, default='/mnt/Dataset/1_views/13_sketch_train_picture')
parser.add_argument('--val-sketch-datadir', type=str, default='/mnt/Dataset/1_views/13_sketch_test_picture')
# parser.add_argument('--sketch-datadir', type=str, default='/mnt/Dataset/Shrec_14/14_sketch_picture')
# parser.add_argument('--val-sketch-datadir', type=str, default='/mnt/Dataset/Shrec_14/14_sketch_test_picture')
parser.add_argument('--view-datadir', type=str, default='/mnt/Dataset/Shrec_14/1_view/14_view_render_img')
#parser.add_argument('--view-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\1_views\\13_view_render_img')
parser.add_argument('--workers', default=4, type=int,help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--sketch-batch-size', type=int, default=64)
parser.add_argument('--view-batch-size', type=int, default=16)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--lr-backbone', type=float, default=4e-4, help="learning rate for backbone")
parser.add_argument('--lr-classifier', type=float, default=4e-3, help="learning rate for classifier")
parser.add_argument('--weight_decay', type=float, default=2e-5)
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--feat-dim', type=int, default=2048, help="feature size")
parser.add_argument('--img-size', type=int, default=224, help="image size")
parser.add_argument('--alph', type=float, default=12, help="L2 alph")

# model
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16', 'vgg19','resnet50','inceptionresnetv2'], default='resnest50')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--loss', type=bool, choices=["cosface", "arcface","mag","ada"], default="cosface")
parser.add_argument('--use-mixup', type=bool, choices=[True, False], default=False, help="mixup")
# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-model-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model-dir', type=str,default='./saved_model/ResNest50_13/cls_uncer')
parser.add_argument('--count', type=int, default=0)

# amsoftmax
parser.add_argument('--margin', default=0.5, type=float,
                    help='lower bound of feature norm')
parser.add_argument('--scale', default=10, type=float,
                    help='lower bound of feature norm')

# magface
parser.add_argument('--l_a', default=10, type=float,
                    help='lower bound of feature norm')
parser.add_argument('--u_a', default=110, type=float,
                    help='upper bound of feature norm')
parser.add_argument('--l_m', default=0.45,
                    type=float, help='low bound of margin')
parser.add_argument('--u_m', default=1.0, type=float,
                    help='the margin slop for m')
parser.add_argument('--lamada', default=10, type=float,
                    help='the lambda for function g')

args = parser.parse_args()
writer = SummaryWriter()


def get_data(sketch_datadir,val_sketch_datadir):
    """Image reading and image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
        transforms.Resize(args.img_size),
        #transforms.RandomRotation(degrees=15),
        #transforms.ColorJitter(),  # Randomly change the brightness, contrast, and saturation of the image
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        #transforms.RandomVerticalFlip(),
        # transforms.RandomCrop(224),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])  # Imagenet standards

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])


    val_sketch_data=datasets.ImageFolder(root=val_sketch_datadir, transform=val_transform)
    val_sketch_dataloaders=DataLoader(val_sketch_data,batch_size=256,num_workers=args.workers)

    sketch_data = datasets.ImageFolder(root=sketch_datadir, transform=image_transforms)
    sketch_dataloaders = DataLoader(sketch_data, batch_size=args.sketch_batch_size, shuffle=True, num_workers=args.workers)


    return sketch_dataloaders,val_sketch_dataloaders

def val(sketch_model,classifier, val_sketch_dataloader,use_gpu):
    with torch.no_grad():
        sketch_model.eval()
        classifier.eval()
        sketch_size = len(val_sketch_dataloader)
        sketch_dataloader_iter = iter(val_sketch_dataloader)
        total = 0.0
        correct = 0.0
        for batch_idx in range(sketch_size):
            sketch = next(sketch_dataloader_iter)
            sketch_data, sketch_labels = sketch
            
            if use_gpu:
                sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()

            sketch_features = sketch_model.forward(sketch_data)
            feature, logits = classifier.forward(sketch_features)

                    
            _, predicted = torch.max(logits.data, 1)
            total += sketch_labels.size(0)
            correct += (predicted == sketch_labels).sum()

        val_acc = correct.item() / total
        return val_acc

def train_sketch(sketch_model, classifier, criterion_am,optimizer_model,sketch_dataloader, use_gpu):
    sketch_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    sketch_size = len(sketch_dataloader)

    sketch_dataloader_iter = iter(sketch_dataloader)

    for batch_idx in range(sketch_size):
        sketch = next(sketch_dataloader_iter)
        sketch_data, sketch_labels = sketch

        if use_gpu:
            sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()
            
        
        
        if args.use_mixup:
            lam = np.random.beta(1.0,1.0)
            index = torch.randperm(sketch_data.size(0)).cuda()
            sketch_data = lam * sketch_data + (1 - lam) * sketch_data[index, :]
        sketch_features = sketch_model.forward(sketch_data)

        concat_feature = sketch_features
        concat_labels = sketch_labels

        feature, logits = classifier.forward(concat_feature)
#         x_norm=torch.norm(feature, dim=1, keepdim=True)
#         print(x_norm.max().item(), x_norm.min().item())
        x_norm=torch.norm(feature, dim=1, keepdim=True).clamp(args.l_a, args.u_a)
        
        if args.use_mixup:
            cls_loss = lam * criterion_am(logits,x_norm,concat_labels)+(1-lam)*criterion_am(logits,x_norm,concat_labels[index])
        else:
            cls_loss = criterion_am(logits,x_norm,concat_labels)

        loss = cls_loss

        _, predicted = torch.max(logits.data, 1)
        total += concat_labels.size(0)
        correct += (predicted == concat_labels).sum()
        avg_acc = correct.item() / total

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, sketch_size, loss.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        writer.add_scalar("Loss", loss.item(), args.count)
        writer.add_scalar("average accuracy", avg_acc, args.count)

    return avg_acc


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic=True
    use_gpu = torch.cuda.is_available()
    best_acc = 0.0
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating model: {}".format(args.model))

    sketch_model = SketchModel(args.model, args.num_classes)
    sketch_model.cuda()
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)
    classifier.cuda()

    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()
    
#     sketch_model.load_state_dict(torch.load(args.model_dir + '/' + args.model+'_best_baseline_sketch_model'  +  '.pth'))
#     classifier.load_state_dict(torch.load(args.model_dir + '/' + args.model+ '_best_baseline_sketch_classifier' + '.pth'))
    #criterion_am = AMSoftMaxLoss(margin=0.0)
    criterion_am =MagFaceLoss(l_a=args.l_a,u_a=args.u_a,l_m=args.l_m,u_m=args.u_m,lamada=args.lamada)


    optimizer_model = torch.optim.SGD([{"params": sketch_model.parameters(),"lr":args.lr_backbone},
                                       {"params": classifier.parameters(),"lr":args.lr_classifier}], momentum=0.9, weight_decay=2e-5)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model,T_max=args.max_epoch, last_epoch=-1)

    sketch_trainloader,val_sketch_dataloader= get_data(args.sketch_datadir,args.val_sketch_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        avg_acc = train_sketch(sketch_model, classifier,criterion_am,optimizer_model, sketch_trainloader, use_gpu)

        val_acc=val(sketch_model,classifier, val_sketch_dataloader,use_gpu)
        print("\tVal Accuracy: %.4f" % (val_acc))
        print("\tBest Val Accuracy: %.4f" % (best_acc))

        if val_acc>best_acc and val_acc>0.8:
            best_acc=val_acc
            torch.save(sketch_model.state_dict(),
                       args.model_dir + '/' +args.model+'_best_baseline_sketch_model' + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' +args.model+ '_best_baseline_sketch_classifier' + '.pth')

        scheduler.step()
    writer.close()

if __name__ == '__main__':
    main()