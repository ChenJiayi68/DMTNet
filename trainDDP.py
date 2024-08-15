r""" DMTNet training (validation) code """
import sys
sys.path.insert(0, "../")

import argparse

import torch.optim as optim
import torch.nn as nn
import torch, gc
import time
from tqdm import tqdm, trange
import torch.distributed as dist
from model.dmtnet import DMTNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.datasetDDP import FSSDataset


def train(epoch, model, dataloader, optimizer, training):
    r""" Train DMTNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    train_bar=tqdm(dataloader,file=sys.stdout,desc='Processing epoch{}'.format(epoch))
    l=len(dataloader)

    # if training:
    for idx, batch in enumerate(train_bar):

        # 1. DMTNetworks forward pass
        batch = utils.to_cuda(batch)
        logit_mask, bg_logit_mask, f_pred_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)
        
        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask']) + 0.5 * model.module.pred_mask_loss(f_pred_mask, batch['query_mask']) + model.module.compute_objective(bg_logit_mask, 1.0 - batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, l, epoch, write_batch_idx=50)

        # 4.
        time.sleep(0.1)


    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Dataset')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal'])
    parser.add_argument('--logpath', type=str, default='test_case')
    parser.add_argument('--bsz', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--load', type=str, default='False')
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Model initialization
    model = DMTNetwork(args.backbone)
    Logger.log_params(model)

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn, sampler_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    FSSDataset.initialize(img_size=400, datapath='../Dataset')
    dataloader_val, sampler_val = FSSDataset.build_dataloader('fss', args.bsz, args.nworker, '0', 'val')
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    map_location = 'cuda:%d' % local_rank
    
    # Load trained model
    if args.load == 'True':
        path = './logs/test_case.log/best_model.pt'
        src_dict = torch.load(path, map_location={'cuda:0': map_location})
        # src_dict = torch.load(path, map_location='cuda:0')
        model.load_state_dict(src_dict, strict=False)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_lr = args.lr
    for epoch in range(args.niter):
        print("Epoch:{}  Lr:{:.2E}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))

        sampler_trn.set_epoch(epoch)

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if (epoch+1) % 5 == 0:
            Logger.save_model_miou(model, epoch, val_miou)

        elif val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)
        
        gc.collect()
        torch.cuda.empty_cache()

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
