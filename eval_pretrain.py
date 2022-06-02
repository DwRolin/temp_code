import argparse
import logging
import os
import pprint
import shutil
import timeit

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from lib import datasets, models
from lib.config import config, update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import validate
from lib.utils.utils import FullModel, create_logger
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")

    parser.add_argument("--cfg", help="experiment configure file name", default="experiments/cityscapes/hrnet_ocr_w18_train_256x128_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml", type=str)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_sampler(dataset):
    from lib.utils.distributed import is_distributed

    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler

        return DistributedSampler(dataset)
    else:
        return None

def eval_pretrain():
    args = parse_args()
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        "writer": SummaryWriter(tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval("datasets." + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        resize=(config.TRAIN.RESIZE[1], config.TRAIN.RESIZE[0]),
        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
        scale_factor=config.TRAIN.SCALE_FACTOR,
    )

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])    
    test_dataset = eval("datasets." + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        resize=(config.TEST.RESIZE[1], config.TEST.RESIZE[0]),
        downsample_rate=1,
    )

    test_sampler = get_sampler(test_dataset)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler,
    )

    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=config.LOSS.OHEMTHRES,
            min_kept=config.LOSS.OHEMKEEP,
            weight=train_dataset.class_weights,
        )
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights)

    model = models.get_seg_model(config)   
    model = FullModel(model, criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # model_state_file = os.path.join(final_output_dir, "deepglobe_fpn.pth")
    model_state_file = "/home/zzh/Project/zzh/MagNet/checkpoints/deepglobe_fpn.pth"
    # model_state_file = "/home/zzh/Project/zzh/MagNet/checkpoints/cityscapes_hrnet.pth"
    checkpoint = torch.load(model_state_file)
    model.module.model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint.items() if k.startswith("model.")}
    ) 

    valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
    msg = "Loss: {:.3f}, MeanIU: {: 4.4f}".format(valid_loss, mean_IoU)
    logging.info(msg)
    logging.info(IoU_array)

if __name__ == "__main__":
    eval_pretrain()