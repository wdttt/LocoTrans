from __future__ import print_function
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import ModelNet40
from model_equi import IECNN_cls_L, knn
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


def _init_():
    if not os.path.exists('results/cls'):
        os.makedirs('results/cls')
    if not os.path.exists('results/cls/' + args.exp_name):
        os.makedirs('results/cls/' + args.exp_name)
    if not os.path.exists('results/cls/' + args.exp_name + '/' + 'models'):
        os.makedirs('results/cls/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls_l.py results/cls' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model_equi.py results/cls' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py results/cls' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py results/cls' + '/' + args.exp_name + '/' + 'data.py.backup')
def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    return feature
def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, shift=args.shift), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'iecnn':
        model = IECNN_cls_L(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0
    best_test_acc_vn = 0
    best_test_acc_1 = 0
    for epoch in range(args.epochs):

        scheduler.step()
        ####################
        # Train
        ####################
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        train_loss_3 = 0.0
        train_loss_4 = 0.0
        count = 0.0
        model.train()
        train_pred_1 = []
        train_pred_2 = []
        train_true = []
        for data, label in tqdm(train_loader):
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(data.shape[0]), device=device)

            data, label = data.to(device), label.to(device).squeeze()
            if trot is not None:
                data = trot.transform_points(data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits, logits_vn, [vector,idx] = model(data)
            loss_1 = criterion(logits, label)
            loss_2 = criterion(logits_vn, label)
            vector = F.normalize(vector, dim=1).permute(0, 3, 2, 1)
            v1 = vector[:, :, 0, :]
            v2 = vector[:, :, 1, :]
            dot_product = (v1 * v2).sum(-1)
            zero_vector = torch.zeros([batch_size, dot_product.size()[1]], device=dot_product.device)
            loss_3 = F.mse_loss(dot_product, zero_vector)

            v1_local = get_graph_feature(v1.transpose(2, 1), idx=idx, k=args.k)
            v2_local = get_graph_feature(v2.transpose(2, 1), idx=idx, k=args.k)
            v1_consist = torch.matmul(v1_local, v1_local.transpose(3, 2)).reshape(-1, args.k)
            v2_consist = torch.matmul(v2_local, v2_local.transpose(3, 2)).reshape(-1, args.k)
            mask = torch.where(v1_consist.mean(dim=-1) > v2_consist.mean(dim=-1), True, False)
            loss_4 = F.mse_loss(v1_consist[mask].detach(), v2_consist[mask]) + F.mse_loss(v1_consist[~mask], v2_consist[~mask].detach())

            loss = loss_1 + loss_2 + args.loss_orth * loss_3 + args.loss_consist * loss_4

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            preds_vn = logits_vn.max(dim=1)[1]
            count += batch_size
            train_loss_1 += loss_1.item() * batch_size
            train_loss_2 += loss_2.item() * batch_size
            train_loss_3 += loss_3.item() * batch_size
            train_loss_4 += loss_4.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred_1.append(preds.detach().cpu().numpy())
            train_pred_2.append(preds_vn.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred_1 = np.concatenate(train_pred_1)
        train_pred_2 = np.concatenate(train_pred_2)

        outstr_1 = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                   train_loss_1 * 1.0 / count,
                                                                                   metrics.accuracy_score(
                                                                                       train_true, train_pred_1),
                                                                                   metrics.balanced_accuracy_score(
                                                                                       train_true, train_pred_1))
        outstr_2 = 'Train %d, loss vn: %.6f, train acc vn: %.6f, train avg acc vn: %.6f' % (epoch,
                                                                                         train_loss_2 * 1.0 / count,
                                                                                         metrics.accuracy_score(
                                                                                             train_true, train_pred_2),
                                                                                         metrics.balanced_accuracy_score(
                                                                                             train_true, train_pred_2))
        outstr_3 = 'Train %d, orth_loss: %.6f,consist_loss: %.6f' % (epoch, train_loss_3 * 1.0 / count, train_loss_4 * 1.0 / count)
        io.cprint(outstr_1)
        io.cprint(outstr_2)
        io.cprint(outstr_3)
        ####################
        # Test
        ####################
        with torch.no_grad():
            test_loss_1 = 0.0
            test_loss_2 = 0.0
            test_loss_3 = 0.0
            count = 0.0
            model.eval()
            test_pred_1 = []
            test_pred_2 = []
            test_pred_3 = []
            test_true = []
            for data, label in tqdm(test_loader):

                trot = None
                if args.rot == 'z':
                    trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True, device=device)
                elif args.rot == 'so3':
                    trot = Rotate(R=random_rotations(data.shape[0]), device=device)
                data, label = data.to(device), label.to(device).squeeze()
                if trot is not None:
                    data = trot.transform_points(data)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits, logits_vn, _ = model(data)

                loss_1 = criterion(logits, label)
                loss_2 = criterion(logits_vn, label)
                preds = logits.max(dim=1)[1]
                preds_vn = logits_vn.max(dim=1)[1]
                preds_1 = (logits + logits_vn).max(dim=1)[1]
                count += batch_size
                test_loss_1 += loss_1.item() * batch_size
                test_loss_2 += loss_2.item() * batch_size
                test_loss_3 += loss_3.item() * batch_size

                test_true.append(label.cpu().numpy())
                test_pred_1.append(preds.detach().cpu().numpy())
                test_pred_2.append(preds_vn.detach().cpu().numpy())
                test_pred_3.append(preds_1.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred_1 = np.concatenate(test_pred_1)
            test_pred_2 = np.concatenate(test_pred_2)
            test_pred_3 = np.concatenate(test_pred_3)
            test_acc = metrics.accuracy_score(test_true, test_pred_1)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred_1)
            test_acc_vn = metrics.accuracy_score(test_true, test_pred_2)
            avg_per_class_acc_vn = metrics.balanced_accuracy_score(test_true, test_pred_2)
            test_acc_1 = metrics.accuracy_score(test_true, test_pred_3)
            avg_per_class_acc_1 = metrics.balanced_accuracy_score(test_true, test_pred_3)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
            if test_acc_vn >= best_test_acc_vn:
                best_test_acc_vn = test_acc_vn
            if test_acc_1 >= best_test_acc_1:
                best_test_acc_1 = test_acc_1
            outstr_1 = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                                                         test_loss_1 * 1.0 / count,
                                                                                                         test_acc,
                                                                                                         avg_per_class_acc,
                                                                                                         best_test_acc)
            outstr_2 = 'Test %d, loss vn: %.6f, test acc vn: %.6f, test avg acc vn: %.6f, best test acc vn: %.6f' % (epoch,
                                                                                                         test_loss_2 * 1.0 / count,
                                                                                                         test_acc_vn,
                                                                                                         avg_per_class_acc_vn,
                                                                                                         best_test_acc_vn)
            outstr_3 = 'Test %d, test acc 1: %.6f, test avg acc 1: %.6f, best test acc 1: %.6f' % (epoch, test_acc_1,
                                                                                              avg_per_class_acc_1,
                                                                                              best_test_acc_1)
            io.cprint(outstr_1)
            io.cprint(outstr_2)
            io.cprint(outstr_3)
            if test_acc >= best_test_acc:
                torch.save(model.state_dict(), 'results/cls/%s/models/model.t7' % args.exp_name)
            if test_acc_vn >= best_test_acc_vn:
                torch.save(model.state_dict(), 'results/cls/%s/models/model_vn.t7' % args.exp_name)
            if test_acc_1 >= best_test_acc_1:
                torch.save(model.state_dict(), 'results/cls/%s/models/model_1.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'iecnn':
        model = IECNN_cls_L(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    model = nn.DataParallel(model)
    if args.model_path is not None:
        model_path = args.model_path
    else:
        #select model_1.t7 or model_2.t7 to load
        model_path = os.path.join('results/cls', args.exp_name, f'models/{args.checkpoint}.t7')
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_true = []
    test_pred_1 = []
    test_pred_2 = []
    test_pred_3 = []
    for data, label in tqdm(test_loader):
        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True, device=device)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(data.shape[0]), device=device)
        data, label = data.to(device), label.to(device).squeeze()
        if trot is not None:
            data = trot.transform_points(data)
        data = data.permute(0, 2, 1)
        logits, logits_vn, _ = model(data)
        preds = logits.max(dim=1)[1]
        preds_vn = logits_vn.max(dim=1)[1]
        preds_1 = (logits+logits_vn).max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred_1.append(preds.detach().cpu().numpy())
        test_pred_2.append(preds_vn.detach().cpu().numpy())
        test_pred_3.append(preds_1.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred_1 = np.concatenate(test_pred_1)
    test_pred_2 = np.concatenate(test_pred_2)
    test_pred_3 = np.concatenate(test_pred_3)
    test_acc = metrics.accuracy_score(test_true, test_pred_1)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred_1)
    test_acc_vn = metrics.accuracy_score(test_true, test_pred_2)
    avg_per_class_acc_vn = metrics.balanced_accuracy_score(test_true, test_pred_2)
    test_acc_1 = metrics.accuracy_score(test_true, test_pred_3)
    avg_per_class_acc_1 = metrics.balanced_accuracy_score(test_true, test_pred_3)
    outstr_1 = 'Test, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (test_acc, avg_per_class_acc, test_acc)
    outstr_2 = 'Test, test acc vn: %.6f, test avg acc vn: %.6f, best test acc vn: %.6f' % (test_acc_vn, avg_per_class_acc_vn, test_acc_vn)
    outstr_3 = 'Test, test acc 1: %.6f, test avg acc 1: %.6f, best test acc 1: %.6f' % (test_acc_1, avg_per_class_acc_1, test_acc_1)
    io.cprint(outstr_1)
    io.cprint(outstr_2)
    io.cprint(outstr_3)



if __name__ == "__main__":
    # Training settings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='modelnet40_cls', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='iecnn', metavar='N',
                        choices=['iecnn'],
                        help='Model to use, [iecnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--checkpoint', type=str, default='model_1', metavar='N',
                        choices=['model', 'model_vn', 'model_1'],
                        help='checkpoint to load')
    parser.add_argument('--rot', type=str, default='z', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    parser.add_argument('--loss_orth', type=float, default=0.1, metavar='N',
                        help='Orthogonality loss.')
    parser.add_argument('--loss_consist', type=float, default=0.1, metavar='N',
                        help='Consistency loss.')
    parser.add_argument('--shift', type=float, default=0.15,
                        help='shift in data augmentation')
    parser.add_argument('--add_layer', type=bool, default=False,
                        help='add one layer in PaRI-Conv')
    args = parser.parse_args()

    _init_()

    io = IOStream('results/cls/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)