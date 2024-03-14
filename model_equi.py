import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
EPS = 1e-6
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:          # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_ = deepcopy(idx)
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature,idx_


def get_graph_feature_(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if num_dims == 3:
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 2*num_dims, num_points, k)

class PaRIConv(nn.Module):
    # modified from https://github.com/GostInShell/PaRI-Conv/blob/main/Networks/model.py
    def __init__(self, in_dim, out_dim, feat_dim=4, k=20,add_layer=False):
        super(PaRIConv, self).__init__()
        self.k = k

        self.basis_matrix = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.dynamic_kernel = nn.Sequential(nn.Conv2d(feat_dim, in_dim//2, kernel_size=1),
                                            nn.BatchNorm2d(in_dim//2),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_dim//2, in_dim, kernel_size=1))
        if add_layer:
            self.act = nn.Sequential(nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True), nn.Conv2d(in_dim, in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True))

        else:
            self.act = nn.Sequential(nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_dim),
                                       nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x, PPF, edge_index, bs):
        _, C = PPF.size()
        PPF = PPF.view(bs, -1, self.k, C).permute(0, 3, 1, 2).contiguous()
        row, col = edge_index
        feat = self.act(self.dynamic_kernel(PPF) * feat_select(self.basis_matrix(x), col))
        pad_x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
        return self.edge_conv(torch.cat((feat - pad_x, pad_x), dim=1)).max(dim=-1, keepdim=False)[0]

def feat_select(feat, ind):
    assert feat.dim()==3 and ind.dim()==1
    B, C, N = feat.size()
    BNK = ind.size(0)
    K = int(BNK/(B*N))
    feat = feat.transpose(2,1).reshape(B*N, -1)[ind, :].reshape(B,N,K,-1).permute(0,3,1,2)
    return feat

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out

class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.out= out_channels
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out

class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)
        if self.normalize_frame:
            self.vn3 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                         negative_slope=negative_slope)
            self.vn4 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                         share_nonlinearity=share_nonlinearity,
                                         negative_slope=negative_slope)
            self.vn_lin2 = nn.Linear(in_channels // 4, 2, bias=False)
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        if self.normalize_frame:
            z1 = self.vn3(z0.detach())
            z1 = self.vn4(z1)
            z1 = self.vn_lin2(z1.transpose(1, -1)).transpose(1, -1)
            z2 = z1.transpose(2,1)

            v1 = self.get_norm(z1[:,0,:])
            v2 = self.get_norm(z1[:, 1, :])
            v = self.get_v(v1,v2)
            u1 = self.get_norm(v-v1)
            u2 = self.get_norm(v-v2)
            z1 = torch.stack([u1, u2, torch.cross(u1, u2)], dim=1).transpose(1, 2)
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        if self.normalize_frame:
            return x_std, z0, z1,z2
        else:
            return x_std, z0
    def get_v(self,v1,v2):
        cos_theta = (v1 * v2).sum(1,keepdims=True).detach()
        cos_theta = torch.clamp(cos_theta,-1+EPS,1-EPS)
        sin_theta_ = torch.sqrt((1 - cos_theta) / 2)
        cos_theta_ = torch.sqrt((1 + cos_theta) / 2)
        v = self.get_norm(v1 + v2)*(sin_theta_ + cos_theta_).repeat(1,3,1)
        return v
    def get_norm(self,v):
        v_norm = torch.sqrt((v * v).sum(1, keepdims=True))
        v = v / (v_norm + EPS)
        return v


class IECNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(IECNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        self.add_layer = args.add_layer
        self.conv1_vn = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2_vn = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3_vn = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        self.conv4_vn = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)
        self.conv5_vn = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True)
        self.std_feature_vn = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=True)
        self.linear1_vn = nn.Linear((1024 // 3) * 12, 512)
        self.bn1_vn = nn.BatchNorm1d(512)
        self.dp1_vn = nn.Dropout(p=args.dropout)
        self.linear2_vn = nn.Linear(512, 256)
        self.bn2_vn = nn.BatchNorm1d(256)
        self.dp2_vn = nn.Dropout(p=args.dropout)
        self.linear3_vn = nn.Linear(256, output_channels)

        if args.pooling == 'max':
            self.pool1_vn = VNMaxPool(64 // 3)
            self.pool2_vn = VNMaxPool(64 // 3)
            self.pool3_vn = VNMaxPool(128 // 3)
            self.pool4_vn = VNMaxPool(256 // 3)
        elif args.pooling == 'mean':
            self.pool1_vn = mean_pool
            self.pool2_vn = mean_pool
            self.pool3_vn = mean_pool
            self.pool4_vn = mean_pool

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pari_2 = PaRIConv(64, 64, k=self.k,feat_dim=63,add_layer=self.add_layer)
        self.pari_3 = PaRIConv(64, 128, k=self.k,feat_dim=63,add_layer=self.add_layer)
        self.pari_4 = PaRIConv(128, 256, k=self.k,feat_dim=126,add_layer=self.add_layer)
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.fc_fuse1 = nn.Linear(1024, 512)
        self.fc_fuse2 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.linear_fuse1 = nn.Linear(512, 256)
        self.bn_fuse1 = nn.BatchNorm1d(256)
        self.dp_fuse1 = nn.Dropout(p=args.dropout)
        self.linear_fuse2 = nn.Linear(256, output_channels)

    def forward(self, x):
        x_vn = copy.deepcopy(x)
        batch_size = x.size(0)
        x_vn = x_vn.unsqueeze(1)
        x_vn,idx1= get_graph_feature(x_vn, k=self.k)
        x_vn = self.conv1_vn(x_vn)
        x1_vn = self.pool1_vn(x_vn)

        x_vn,idx2 = get_graph_feature(x1_vn, k=self.k)
        x_vn = self.conv2_vn(x_vn)
        x2_vn = self.pool2_vn(x_vn)

        x_vn,idx3 = get_graph_feature(x2_vn, k=self.k)
        x_vn = self.conv3_vn(x_vn)
        x3_vn = self.pool3_vn(x_vn)

        x_vn,idx4 = get_graph_feature(x3_vn, k=self.k)
        x_vn = self.conv4_vn(x_vn)
        x4_vn = self.pool4_vn(x_vn)

        x_vn = torch.cat((x1_vn, x2_vn, x3_vn, x4_vn), dim=1)
        x_vn = self.conv5_vn(x_vn)
        num_points = x_vn.size(-1)
        x_vn_mean = x_vn.mean(dim=-1, keepdim=True).expand(x_vn.size())
        x_vn = torch.cat((x_vn, x_vn_mean), 1)
        x_vn, trans_vn, trans_vn_normalize,vector= self.std_feature_vn(x_vn)
        x_vn = x_vn.view(batch_size, -1, num_points)

        x = get_graph_feature_(x, k=self.k,idx=idx1)
        x[:,:3,:,:] = torch.einsum('bijk,bimj->bmjk', x[:,:3,:,:], trans_vn_normalize)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        eq_feat, edge_index = self.get_equi_feature(x1 ,x1_vn,trans_vn_normalize.detach(),idx=idx2)
        x2 = self.pari_2(x1, eq_feat, edge_index, bs=batch_size)

        eq_feat, edge_index = self.get_equi_feature(x2,x2_vn, trans_vn_normalize.detach(),idx=idx3)
        x3 = self.pari_3(x2, eq_feat, edge_index, bs=batch_size)

        eq_feat, edge_index = self.get_equi_feature(x3,x3_vn,trans_vn_normalize.detach(),idx=idx4)
        x4 = self.pari_4(x3, eq_feat, edge_index, bs=batch_size)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1_vn = F.adaptive_max_pool1d(x_vn, 1).view(batch_size, -1)
        x2_vn = F.adaptive_avg_pool1d(x_vn, 1).view(batch_size, -1)
        x_vn = torch.cat((x1_vn, x2_vn), 1)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x_vn = F.leaky_relu(self.bn1_vn(self.linear1_vn(x_vn)), negative_slope=0.2)
        x_vn = self.dp1_vn(x_vn)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)

        x_cat = self.fc_fuse1(torch.cat([x, x_vn], dim=1))
        x_weight = torch.sigmoid(self.fc_fuse2(x_cat))
        x_fuse = F.relu(x_cat * x_weight)
        x_fuse = F.leaky_relu(self.bn_fuse1(self.linear_fuse1(x_fuse)), negative_slope=0.2)
        x_fuse = self.dp_fuse1(x_fuse)
        x_fuse = self.linear_fuse2(x_fuse)

        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x_vn = F.leaky_relu(self.bn2_vn(self.linear2_vn(x_vn)), negative_slope=0.2)
        x_vn = self.dp2_vn(x_vn)
        x_vn = self.linear3_vn(x_vn)
        return x,x_vn,x_fuse,[vector,idx1]
    def get_equi_feature(self, x, x_vn,trans, idx=None):
        if idx is None:
            idx = knn(x, k=self.k).cuda()  # (batch_size, num_points, k)

        batch_size = idx.size(0)
        num_points = idx.size(1)

        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        col = idx.view(-1)
        row = (torch.arange(num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, self.k) + idx_base).view(-1)
        bs, num_dims,_, _ = x_vn.size()
        x_vn = x_vn.permute(0,3,1,2)
        feature = x_vn.reshape(batch_size * num_points, num_dims,3)[idx,:]
        feature = feature.view(batch_size, num_points, self.k, num_dims,3)
        x_vn = x_vn.view(batch_size, num_points, 1, num_dims,3).repeat(1,1,self.k,1,1)
        eq_feat = (feature - x_vn).permute(0, 3, 4, 1, 2).contiguous()
        eq_feat = torch.einsum('blijk,bimj->blmjk', eq_feat, trans)
        eq_feat = eq_feat.permute(0,3,4,1,2).reshape(batch_size*num_points*self.k,-1)

        return eq_feat, [row, col]

class IECNN_cls_L(nn.Module):
    def __init__(self, args, output_channels=40):
        super(IECNN_cls_L, self).__init__()
        self.args = args
        self.k = args.k
        self.add_layer = args.add_layer
        self.conv1_vn = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2_vn = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3_vn = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4_vn = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv5_vn = VNLinearLeakyReLU(64 // 3 + 64 // 3 + 64 // 3 * 2, 256 // 3, dim=4, share_nonlinearity=True)
        self.std_feature_vn = VNStdFeature(256 // 3, dim=4, normalize_frame=True)
        self.linear1_vn = nn.Linear((256 // 3) * 6, 512)
        self.bn1_vn = nn.BatchNorm1d(512)
        self.dp1_vn = nn.Dropout(p=args.dropout)
        self.linear2_vn = nn.Linear(512, 256)
        self.bn2_vn = nn.BatchNorm1d(256)
        self.dp2_vn = nn.Dropout(p=args.dropout)
        self.linear3_vn = nn.Linear(256, output_channels)

        if args.pooling == 'max':
            self.pool1_vn = VNMaxPool(64 // 3)
            self.pool2_vn = VNMaxPool(64 // 3)
            self.pool3_vn = VNMaxPool(128 // 3)
            self.pool4_vn = VNMaxPool(256 // 3)
        elif args.pooling == 'mean':
            self.pool1_vn = mean_pool
            self.pool2_vn = mean_pool
            self.pool3_vn = mean_pool
            self.pool4_vn = mean_pool

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pari_2 = PaRIConv(64, 64, k=self.k,feat_dim=63,add_layer=self.add_layer)
        self.pari_3 = PaRIConv(64, 128, k=self.k,feat_dim=63,add_layer=self.add_layer)
        self.pari_4 = PaRIConv(128, 256, k=self.k,feat_dim=63,add_layer=self.add_layer)
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x_vn = copy.deepcopy(x)
        batch_size = x.size(0)
        x_vn = x_vn.unsqueeze(1)
        x_vn,idx1= get_graph_feature(x_vn, k=self.k)
        x_vn = self.conv1_vn(x_vn)
        x1_vn = self.pool1_vn(x_vn)

        x_vn,idx2 = get_graph_feature(x1_vn, k=self.k)
        x_vn = self.conv2_vn(x_vn)
        x2_vn = self.pool2_vn(x_vn)

        x_vn,idx3 = get_graph_feature(x2_vn, k=self.k)
        x_vn = self.conv3_vn(x_vn)
        x3_vn = self.pool3_vn(x_vn)

        x_vn,idx4 = get_graph_feature(x3_vn, k=self.k)
        x_vn = self.conv4_vn(x_vn)
        x4_vn = self.pool4_vn(x_vn)

        x_vn = torch.cat((x1_vn, x2_vn, x3_vn, x4_vn), dim=1)
        x_vn = self.conv5_vn(x_vn)
        num_points = x_vn.size(-1)
        x_vn, trans_vn, trans_vn_normalize,vector= self.std_feature_vn(x_vn)
        x_vn = x_vn.view(batch_size, -1, num_points)

        x = get_graph_feature_(x, k=self.k)
        x[:,:3,:,:] = torch.einsum('bijk,bimj->bmjk', x[:,:3,:,:], trans_vn_normalize)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        eq_feat, edge_index = self.get_equi_feature(x1 ,x1_vn,trans_vn_normalize.detach())
        x2 = self.pari_2(x1, eq_feat, edge_index, bs=batch_size)

        eq_feat, edge_index = self.get_equi_feature(x2,x2_vn, trans_vn_normalize.detach())
        x3 = self.pari_3(x2, eq_feat, edge_index, bs=batch_size)

        eq_feat, edge_index = self.get_equi_feature(x3,x3_vn,trans_vn_normalize.detach())
        x4 = self.pari_4(x3, eq_feat, edge_index, bs=batch_size)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1_vn = F.adaptive_max_pool1d(x_vn, 1).view(batch_size, -1)
        x2_vn = F.adaptive_avg_pool1d(x_vn, 1).view(batch_size, -1)
        x_vn = torch.cat((x1_vn, x2_vn), 1)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x_vn = F.leaky_relu(self.bn1_vn(self.linear1_vn(x_vn)), negative_slope=0.2)
        x_vn = self.dp1_vn(x_vn)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x_vn = F.leaky_relu(self.bn2_vn(self.linear2_vn(x_vn)), negative_slope=0.2)
        x_vn = self.dp2_vn(x_vn)
        x_vn = self.linear3_vn(x_vn)
        return x,x_vn,[vector,idx1]
    def get_equi_feature(self, x, x_vn,trans, idx=None):
        if idx is None:
            idx = knn(x, k=self.k).cuda()  # (batch_size, num_points, k)

        batch_size = idx.size(0)
        num_points = idx.size(1)

        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        col = idx.view(-1)
        row = (torch.arange(num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, self.k) + idx_base).view(-1)
        bs, num_dims,_, _ = x_vn.size()
        x_vn = x_vn.permute(0,3,1,2)
        feature = x_vn.reshape(batch_size * num_points, num_dims,3)[idx,:]
        feature = feature.view(batch_size, num_points, self.k, num_dims,3)
        x_vn = x_vn.view(batch_size, num_points, 1, num_dims,3).repeat(1,1,self.k,1,1)
        eq_feat = (feature - x_vn).permute(0, 3, 4, 1, 2).contiguous()
        eq_feat = torch.einsum('blijk,bimj->blmjk', eq_feat, trans)
        eq_feat = eq_feat.permute(0,3,4,1,2).reshape(batch_size*num_points*self.k,-1)

        return eq_feat, [row, col]