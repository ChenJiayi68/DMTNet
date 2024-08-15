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


def test(model, dataloader, nshot):
    r""" Test DMTNet """

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. DMTNetworks forward pass
        batch = utils.to_cuda(batch)
        pred_mask, _, _ = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
        # 3. Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
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
    model.eval()

    # Test DMTNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
