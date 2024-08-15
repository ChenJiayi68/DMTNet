r""" Cross-Domain Few-Shot Semantic Segmentation testing code """
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch
import os
from model.dmtnet import DMTNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.datasetDDP import FSSDataset
from common.vis import Visualizer
import torch.optim as optim
import torch.nn.functional as F


def test(model, dataloader, nshot):
    r""" Test DMTNet """
    to_unfreeze_dict = ['module.hpn_learner.encoder_layer3to2.7.weight','module.hpn_learner.encoder_layer4to3.7.weight']

    for (name,param) in model.named_parameters():
        if name in to_unfreeze_dict:
            pass
        else:
            param.requires_grad = False
    
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        # 1. DMTNetworks forward pass
        batch = utils.to_cuda(batch)
        pred_mask, logit_mask_orig = model.module.predict_mask_nshot_support(batch, nshot)
        assert pred_mask.size() == batch['query_mask'].size()
        
        optimizer.zero_grad()
        loss = model.module.compute_objective_finetuning(logit_mask_orig, batch['support_masks'].clone(), nshot) 
        
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        # for (name,param) in model.named_parameters():
        #     if param.requires_grad == True:
        #         print('name:', name)
        #         print('param.grad:', param.grad)

        # 2. Evaluate prediction
        with torch.no_grad():
            pred_mask_final, _, _ = model.module.predict_mask_nshot(batch, nshot)
            assert pred_mask_final.size() == batch['query_mask'].size()
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_final.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
        
        # 3. Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask_final, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Dataset')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['fss', 'deepglobe', 'isic', 'lung', 'pascal'])
    parser.add_argument('--logpath', type=str, default='./test_logs')
    parser.add_argument('--bsz', type=int, default=30)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='path_to_your_trained_model')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--finetuning', type=str, default='True')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Model initialization
    model = DMTNetwork(args.backbone)
    Logger.log_params(model)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test, sampler = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    path = args.load
    src_dict = torch.load(path, map_location='cuda:0')
    model.load_state_dict(src_dict, strict=False)

    # Test DMTNet
    test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
