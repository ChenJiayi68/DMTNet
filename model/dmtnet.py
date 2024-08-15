r""" DMTNetwork """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
import numpy as np


class DMTNetwork(nn.Module):
    def __init__(self, backbone):
        super(DMTNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.reference_layer3 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
            # query transformation linear weight
            self.reference_layer6 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer6.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer6.bias, 0)
            self.reference_layer5 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer5.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer5.bias, 0)
            self.reference_layer4 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer4.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer4.bias, 0)
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.reference_layer3 = nn.Linear(2048, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(1024, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
            # query transformation linear weight
            self.reference_layer6 = nn.Linear(2048, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer6.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer6.bias, 0)
            self.reference_layer5 = nn.Linear(1024, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer5.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer5.bias, 0)
            self.reference_layer4 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer4.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer4.bias, 0)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            bg_support_feats, prototypes_f_grid, prototypes_b_grid = self.mask_feature_grid(support_feats, support_mask.clone(), query_feats)
            support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())

        prototypes_f_q, prototypes_b_q, pred_mask = self.query_prototypes(query_feats, prototypes_f_grid, prototypes_b_grid)
        query_feats, support_feats, bg_support_feats = self.Transformation_Feature(query_feats, support_feats, bg_support_feats, prototypes_f, prototypes_b, prototypes_f_q, prototypes_b_q)

        corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
        bg_corr = Correlation.multilayer_correlation(query_feats, bg_support_feats, self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)
        bg_logit_mask = self.hpn_learner(bg_corr)
        bg_logit_mask = F.interpolate(bg_logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask, bg_logit_mask, pred_mask
               
    def get_index(self, i, j, k):
        assert i >= j
        for t in range(0, i - j + 1, k):
            yield t
        if t + j < i:
            yield i - j
    
    def get_grid(self, shape, grid_num, overlap=False):
        i_h, i_w = shape
        j_h, j_w = i_h//grid_num,i_w//grid_num
        if overlap == False:
            k_h,k_w = j_h, j_w
        else:
            k_h,k_w = j_h//2, j_w//2
        nums_h = self.get_index(i_h, j_h, k_h)
        grids = []
        for h in nums_h:
            nums_w = self.get_index(i_w, j_w, k_w)
            for w in nums_w:
                grid_idx = (
                        slice(h, h + j_h),
                        slice(w, w + j_w)
                )
                grids.append(grid_idx)
        return grids

    def mask_feature_grid(self, features, support_mask, query_feats):
        eps = 1e-6
        prototypes_f = []
        prototypes_b = []
        bg_features = []
        fg_features = []
        for idx, feature in enumerate(features): # [layernum, batchsize, C, H, W]
            fg_mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            bg_mask = 1 - fg_mask
            bg_features.append(features[idx] * bg_mask)
            fg_features.append(features[idx] * fg_mask)
            grids = self.get_grid(fg_features[idx].shape[-2:], grid_num = 4)
            # prototype
            grid_pf = []
            grid_pb = []
            grid_qf = []
            for grid in grids:
                grid_pf.append(torch.sum(fg_features[idx][:,:,grid[0],grid[1]],dim = (2, 3))/((fg_mask[:,:,grid[0],grid[1]]).sum((2, 3)) + eps))
                grid_pb.append(torch.sum(bg_features[idx][:,:,grid[0],grid[1]],dim = (2, 3))/((bg_mask[:,:,grid[0],grid[1]]).sum((2, 3)) + eps))
                grid_qf.append(query_feats[idx][:,:,grid[0],grid[1]])
            prototypes_f.append(grid_pf)
            prototypes_b.append(grid_pb)
        # prototypes_f [layernum, gridnum, bsz, ch]
        # prototypes_b [layernum, gridnum, bsz, ch]
        return bg_features, prototypes_f, prototypes_b
    
    def mask_feature(self, features, support_mask):
        eps = 1e-6
        prototypes_f = []
        prototypes_b = []
        bg_features = []
        mask_features = []
        for idx, feature in enumerate(features): # [layernum, batchsize, C, H, W]
            # support_mask.shape [bsz, h, w]
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            bg_mask = 1 - mask
            bg_features.append(features[idx] * bg_mask)
            mask_features.append(features[idx] * mask)
            features[idx] = features[idx] * mask
            # prototype
            proto_f = features[idx].sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_f.append(proto_f)
            proto_b = bg_features[idx].sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            prototypes_b.append(proto_b)
        return mask_features, prototypes_f, prototypes_b

    def Transformation_Feature(self, query_feats, support_feats, bg_support_feats, prototypes_f, prototypes_b, prototypes_b_q, prototypes_f_q):
        transformed_query_feats = []
        transformed_support_feats = []
        transformed_bg_support_feats = []
        bsz = query_feats[0].shape[0]
        for idx, feature in enumerate(support_feats):
            C = torch.cat((prototypes_b[idx].unsqueeze(1), prototypes_f[idx].unsqueeze(1)), dim=1)  # C.shape [bsz, 2, ch]
            C_q = torch.cat((prototypes_b_q[idx].unsqueeze(1), prototypes_f_q[idx].unsqueeze(1)), dim=1)  # C_q.shape [bsz, 2, ch]
            eps = 1e-6
            if idx <= 3:
                R = self.reference_layer1.weight.expand(C.shape)
                R_q = self.reference_layer4.weight.expand(C_q.shape)
            elif idx <= 9:
                R = self.reference_layer2.weight.expand(C.shape)
                R_q = self.reference_layer5.weight.expand(C_q.shape)
            elif idx <= 12:
                R = self.reference_layer3.weight.expand(C.shape)
                R_q = self.reference_layer6.weight.expand(C_q.shape)
            R = self.dropout(R)
            R_q = self.dropout(R_q)
            power_R = ((R * R).sum(dim=2, keepdim=True)).sqrt()
            power_R_q = ((R_q * R_q).sum(dim=2, keepdim=True)).sqrt()
            R_q = R_q / (power_R_q + eps)
            R = R / (power_R + eps)
            # R.shape [bsz, 4, ch]
            power_C = ((C * C).sum(dim=2, keepdim=True)).sqrt()
            C = C / (power_C + eps)
            power_C_q = ((C_q * C_q).sum(dim=2, keepdim=True)).sqrt()
            C_q = C_q / (power_C_q + eps)
            # C.shape [bsz, 4, ch]

            P = torch.matmul(torch.pinverse(C), R)
            P = P.permute(0, 2, 1)
            P_q = torch.matmul(torch.pinverse(C_q), R_q)
            P_q = P_q.permute(0, 2, 1)
            beta = 0.5
            P_q = P_q * beta + P * (1.0 - beta)
            # P.shape [bsz, ch, ch]

            init_size = query_feats[idx].shape
            query_feats[idx] = query_feats[idx].view(bsz, C_q.size(2), -1)
            transformed_query_feats.append(torch.matmul(P_q, query_feats[idx]).view(init_size))
            init_size = feature.shape
            feature = feature.view(bsz, C.size(2), -1)
            transformed_support_feats.append(torch.matmul(P, feature).view(init_size))
            init_size = bg_support_feats[idx].shape
            bg_support_feats[idx] = bg_support_feats[idx].view(bsz, C.size(2), -1)
            transformed_bg_support_feats.append(torch.matmul(P, bg_support_feats[idx]).view(init_size))

        return transformed_query_feats, transformed_support_feats, transformed_bg_support_feats

    def calDist(self, feature, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(feature, prototype[..., None, None], dim=1) * scaler
        return dist # dist:[1,53,53]

    def predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_orig = []
        bg_logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask, bg_logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            bg_logit_mask_orig.append(bg_logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig, bg_logit_mask_orig
        # logit_mask_orig.size [bsz,c,w,h]
        # logit_mask_agg.size [bsz,w,h]
        # pred_mask.size [bsz,w,h]
        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig, bg_logit_mask_orig
    
    def predict_mask_nshot_support(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask, _, _ = self(batch['support_imgs'][:, s_idx], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig
        # logit_mask_orig.size [bsz,c,w,h]
        # logit_mask_agg.size [bsz,w,h]
        # pred_mask.size [bsz,w,h]
        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        # logit_mask.shape [20,2,160000]
        # gt_mask.shape [20,160000]

        return self.cross_entropy_loss(logit_mask, gt_mask)
    
    def compute_objective_finetuning(self, logit_mask, gt_mask, nshot):
        # logit_mask [nshot, bsz, c, h, w]
        # gt_mask [bsz, nshot, h, w]
        loss = 0.0
        for idx in range(nshot):
            bsz = gt_mask.shape[0]
            loss += self.cross_entropy_loss(logit_mask[idx].view(bsz, 2, -1), gt_mask[:,idx].view(bsz, -1).long())
        return loss/nshot

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def test_finetune_mode(self,to_unfreeze_dict):
        self.train()
        for (name,param) in self.named_parameters():
            if name in to_unfreeze_dict:
                pass
            else:
                param.requires_grad = False

    def query_prototypes(self, query_feats, prototypes_f, prototypes_b):
        # prototypes_f [layernum, gridnum, bsz, ch]
        # prototypes_b [layernum, gridnum, bsz, ch]
        # query_feats [layernum, bsz, ch, h, w]
        result = []
        for idx, query_feat in enumerate(query_feats):
            bsz,_,h,w = query_feat.shape
            out = torch.zeros(bsz, 2, h, w).float().cuda()
            for i, _ in enumerate(prototypes_f[idx]):
                s_fg = F.cosine_similarity(query_feat, prototypes_f[idx][i].unsqueeze(-1).unsqueeze(-1), dim=1) # [bsz, h, w]
                s_bg = F.cosine_similarity(query_feat, prototypes_b[idx][i].unsqueeze(-1).unsqueeze(-1), dim=1) # [bsz, h, w]
                out = out + torch.cat((s_bg[:, None, ...], s_fg[:, None, ...]), dim=1) * 10.0 # [bsz, 2, h, w]
            out = out/float(len(prototypes_f[0]))
            result.append(out)

        prototypes_f_q = []
        prototypes_b_q = []
        eps = 1e-6
        
        for idx, sim in enumerate(result):
            pred_mask = sim.argmax(dim=1)
            pred_mask = pred_mask.unsqueeze(1).float()
            bg_mask = 1 - pred_mask
            fg_feature = query_feats[idx] * pred_mask
            bg_feature = query_feats[idx] * bg_mask
            proto_f = fg_feature.sum((2, 3))
            label_sum = pred_mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            proto_b = bg_feature.sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b/(label_sum + eps)
            prototypes_f_q.append(proto_f)
            prototypes_b_q.append(proto_b)
       
        return prototypes_f_q, prototypes_b_q, result
      
    def pred_mask_loss(self, pred_mask, gt):
        # pred_mask [layernum, bsz, 2, h, w]
        # gt [bsz, h, w]
        loss =  0.0
        for mask in pred_mask:
            gt_mask = F.interpolate(gt.unsqueeze(1).float(), mask.size()[2:], mode='bilinear', align_corners=True)
            loss = loss + self.compute_objective(mask, gt_mask)
        loss = loss / 13.0
        return loss