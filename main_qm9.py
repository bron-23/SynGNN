import argparse
import datetime
import itertools
import pickle
import subprocess
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader

import e3nn.o3 as o3
from IPython import embed
import os
print(f"--- DEBUG: CUDA_VISIBLE_DEVICES seen by script: {os.environ.get('CUDA_VISIBLE_DEVICES')} ---")
# Rest of your script...

from logger import FileLogger
from pathlib import Path

from datasets.qm9 import QM9
#from torch_geometric.nn import SchNet

# AMP
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import train_one_epoch, evaluate, compute_stats
import torch
import torch_geometric.utils # 我们需要 blacklisted 的函数所在的模块
import torch_cluster        # radius 函数所在的模块
# distributed training
import utils
# ============ 添加这部分代码来处理编译错误 ============
# 检查 PyTorch 版本是否支持 torch.compile
IS_PYTORCH_2_OR_GREATER = True
try:
    from packaging import version
    if version.parse(torch.__version__) < version.parse("2.0.0"):
        IS_PYTORCH_2_OR_GREATER = False
except (ImportError, ModuleNotFoundError):
    # 如果没有 packaging 库，用简单的方式判断
    if int(torch.__version__.split('.')[0]) < 2:
        IS_PYTORCH_2_OR_GREATER = False
# if IS_PYTORCH_2_OR_GREATER:
#     import torch._dynamo
#
#     print("PyTorch >= 2.0.0 detected. Blacklisting problematic functions for torch.compile.")
#
#     # 将 torch_cluster.radius_graph 加入黑名单
#     # 这是 torch_geometric.radius_graph 的真实来源
#     if hasattr(torch_cluster, 'radius_graph'):
#         torch._dynamo.disallow_in_graph(torch_cluster.radius_graph)
#
#     # 你的代码中还调用了 torch_geometric.nn.radius
#     # 其底层是 torch_cluster.radius，所以我们把它也加入黑名单
#     if hasattr(torch_cluster, 'radius'):
#         torch._dynamo.disallow_in_graph(torch_cluster.radius)
#
#     # 如果你的模型里直接使用了 torch_geometric.nn.radius，也需要处理它
#     # 通常处理底层的 torch_cluster 已经足够，但为了保险可以都加上
#     if hasattr(torch_geometric.nn, 'radius'):
#         torch._dynamo.disallow_in_graph(torch_geometric.nn.radius)
#
# # =======================================================
ModelEma = ModelEmaV2


def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # --- 在这里添加新的参数 ---
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # network architecture
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--num-basis', type=int, default=32)
    parser.add_argument('--output-channels', type=int, default=1)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--data-path", type=str, default='data/qm9')
    parser.add_argument('--feature-type', type=str, default='one_hot')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='l1')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # AMP
    parser.add_argument('--no-amp', action='store_false', dest='amp', 
                        help='Disable FP16 training.')
    parser.set_defaults(amp=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # EMPP self-supervisied learning
    parser.add_argument('--ssp', action='store_true', 
                        help='using EMPP as auxiliary tasks')

    # EMPP self-supervised learning
    parser.add_argument('--empp-loss-weight', type=float, default=1.0,  # New or updated
                        help='Weight for the EMPP loss component')
    parser.add_argument('--empp-num-mask', type=int, default=1,  # New
                        help='Number of atoms to mask simultaneously per molecule for EMPP')
    parser.add_argument('--empp-ssp-feature-dim', type=str, default='32x0e',  # New
                        help='Irreps string for the SSP feature injected in TransBlock (e.g., "32x0e")')
    parser.add_argument('--empp-atom-type-embed-irreps', type=str, default='16x0e',  # New
                        help='Irreps string for embedding masked atom types for SSP injection (e.g., "32x0e")')
    # Optional: Control heteroatom prioritization via flag
    parser.add_argument('--empp-prioritize-heteroatoms', action='store_true', default=True,  # New, default to True
                        help='Prioritize masking heteroatoms for EMPP.')
    parser.add_argument('--empp-no-prioritize-heteroatoms', action='store_false', dest='empp_prioritize_heteroatoms',
                        help='Disable heteroatom prioritization for EMPP.')

    # New argument for pos_prediction's num_s2_channels
    parser.add_argument('--empp-pos-pred-num-s2-channels', type=int, default=32,  # New
                        help='Number of channels for S2 projection in EMPP position prediction head.')

    # res_s2grid, temperature_softmax 等 pos_prediction 的其他参数也可以按需添加。
    parser.add_argument('--empp-pos-pred-res-s2grid', type=int, default=100,
                        help='Resolution for S2 grid in pos_prediction.')
    parser.add_argument('--empp-pos-pred-temp-softmax', type=float, default=0.1,
                        help='Temperature for softmax in pos_prediction direction output.')
    parser.add_argument('--empp-pos-pred-temp-label', type=float, default=0.1,
                        help='Temperature for softmax of ground truth direction in pos_prediction.')

    # --- Contrastive Learning ---
    parser.add_argument('--enable-contrastive', action='store_true', default=False,  # Explicit default
                        help='Enable Contrastive Learning auxiliary task.')
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.1,
                        help='Weight for the Contrastive loss component.')
    parser.add_argument('--contrastive-projection-dim', type=int, default=128,
                        help='Output dimension of the contrastive projection head.')
    parser.add_argument('--contrastive-temp', type=float, default=0.1,
                        help='Temperature scaling for InfoNCE loss in contrastive learning.')
    parser.add_argument('--contrastive-aug-mask-ratio', type=float, default=0.15,
                        help='Ratio of atom features to mask for contrastive learning augmentation.')
    parser.add_argument('--contrastive-mask-token-idx', type=int, default=0,
                        # Example: use index 0 (e.g. Carbon) as mask
                        help='Index of the token used for masking atom types in contrastive augmentation. Ensure your NodeEmbeddingNetwork handles this or use a dedicated mask embedding.')

    # ============ 在这里添加新参数的定义 ============
    parser.add_argument('--contrastive-prioritize-heteroatoms', action='store_true',
                        help='Enable heteroatom-prioritized masking for contrastive learning augmentation.')
    # ============================================
    return parser


def main(args):
    # ==================== 1. 初始化分布式环境 ====================
    utils.init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================== 2. 初始化日志和随机种子 ====================
    # 日志只由主进程创建和写入
    _log = None
    if utils.is_main_process():
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    args.logger = _log

    if utils.is_main_process():
        _log.info(f"Full training arguments: {args}")

    # 为每个进程设置不同的随机种子，确保初始权重和数据增强不同
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ==================== 3. 加载数据集 ====================
    # 所有进程都需要知道数据集的结构
    train_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    val_dataset = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset = QM9(args.data_path, 'test', feature_type=args.feature_type)

    # 标准化因子只在主进程计算一次即可，或所有进程计算一次也无妨
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]

    if utils.is_main_process():
        _log.info('Training set mean: {}, std:{}'.format(norm_factor[0], norm_factor[1]))

    # ==================== 4. 创建分布式数据加载器 ====================
    sampler_train = DistributedSampler(train_dataset, shuffle=True) if args.distributed else RandomSampler(
        train_dataset)
    sampler_val = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    sampler_test = DistributedSampler(test_dataset, shuffle=False) if args.distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler_val,
                            num_workers=args.workers, pin_memory=args.pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=sampler_test,
                             num_workers=args.workers, pin_memory=args.pin_mem)

    # ==================== 5. 创建并包装模型 ====================
    create_model = model_entrypoint(args.model_name)
    model = create_model(**vars(args))  # 将所有args作为关键字参数传递
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    if utils.is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log.info('Number of params: {}'.format(n_parameters))
        _log.info(f'Model: {model_without_ddp}')

    model_ema = ModelEmaV2(model_without_ddp, decay=args.model_ema_decay) if args.model_ema else None

    # ==================== 6. 创建优化器, 调度器, 损失函数 ====================
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

    # ==================== 7. 加载检查点 (恢复训练) ====================
    start_epoch = 0
    best_val_err, best_test_err, best_train_err, best_epoch = float('inf'), float('inf'), float('inf'), 0
    best_ema_val_err, best_ema_test_err, best_ema_epoch = float('inf'), float('inf'), 0

    if args.resume:
        if os.path.isfile(args.resume):
            if utils.is_main_process(): _log.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')  # 先加载到CPU

            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

            if model_ema and 'model_ema' in checkpoint: model_ema.load_state_dict(checkpoint['model_ema'])
            if 'best_val_err' in checkpoint: best_val_err = checkpoint['best_val_err']
            if 'best_test_err' in checkpoint: best_test_err = checkpoint['best_test_err']
            if 'best_train_err' in checkpoint: best_train_err = checkpoint['best_train_err']
            if 'best_epoch' in checkpoint: best_epoch = checkpoint['best_epoch']
            if 'best_ema_val_err' in checkpoint: best_ema_val_err = checkpoint['best_ema_val_err']
            if 'best_ema_test_err' in checkpoint: best_ema_test_err = checkpoint['best_ema_test_err']
            if 'best_ema_epoch' in checkpoint: best_ema_epoch = checkpoint['best_ema_epoch']

            if utils.is_main_process(): _log.info(
                f"Resumed from epoch {checkpoint['epoch']}. Best val MAE so far: {best_val_err:.5f}")
        else:
            if utils.is_main_process(): _log.info(f"No checkpoint found at '{args.resume}'")

    # ==================== 8. AMP和训练循环 ====================
    loss_scaler = NativeScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress

    if utils.is_main_process(): _log.info(f"Start training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_err = train_one_epoch(
            model=model, criterion=criterion, norm_factor=norm_factor,
            data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=loss_scaler,
            amp_autocast=amp_autocast, model_ema=model_ema, args=args
        )

        lr_scheduler.step(epoch)

        # 在评估前同步所有进程，确保所有训练步骤都已完成
        if args.distributed:
            torch.distributed.barrier()

        val_err, _ = evaluate(model, norm_factor, args.target, val_loader, device, amp_autocast, args)
        test_err, _ = evaluate(model, norm_factor, args.target, test_loader, device, amp_autocast, args)

        is_best = False
        if val_err < best_val_err:
            best_val_err, best_test_err, best_train_err, best_epoch = val_err, test_err, train_err, epoch
            is_best = True

        if utils.is_main_process():
            _log.info(
                f"Epoch: [{epoch}] Target: [{args.target}] train MAE: {train_err:.5f}, val MAE: {val_err:.5f}, test MAE: {test_err:.5f}")
            _log.info(
                f"Best -- epoch={best_epoch}, train MAE: {best_train_err:.5f}, val MAE: {best_val_err:.5f}, test MAE: {best_test_err:.5f}")

        is_best_ema = False
        if model_ema:
            ema_val_err, _ = evaluate(model_ema.module, norm_factor, args.target, val_loader, device, amp_autocast,
                                      args)
            ema_test_err, _ = evaluate(model_ema.module, norm_factor, args.target, test_loader, device, amp_autocast,
                                       args)

            if ema_val_err < best_ema_val_err:
                best_ema_val_err, best_ema_test_err, best_ema_epoch = ema_val_err, ema_test_err, epoch
                is_best_ema = True

            if utils.is_main_process():
                _log.info(f"Epoch: [{epoch}] EMA val MAE: {ema_val_err:.5f}, EMA test MAE: {ema_test_err:.5f}")
                _log.info(
                    f"Best EMA -- epoch={best_ema_epoch}, val MAE: {best_ema_val_err:.5f}, test MAE: {best_ema_test_err:.5f}\n")

        # ==================== 9. 保存检查点 (只在主进程) ====================
        if utils.is_main_process():
            checkpoint_data = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_val_err': best_val_err, 'best_test_err': best_test_err,
                'best_train_err': best_train_err, 'best_epoch': best_epoch,
            }
            if model_ema:
                checkpoint_data['model_ema'] = model_ema.state_dict()
                checkpoint_data['best_ema_val_err'] = best_ema_val_err
                checkpoint_data['best_ema_test_err'] = best_ema_test_err
                checkpoint_data['best_ema_epoch'] = best_ema_epoch

            latest_path = os.path.join(args.output_dir, 'latest.pth')
            torch.save(checkpoint_data, latest_path)

            # if (model_ema and is_best_ema) or (not model_ema and is_best):
            #     best_path = os.path.join(args.output_dir, 'best.pth')
            #     torch.save(checkpoint_data, best_path)

    if utils.is_main_process():
        _log.info("Training finished.")
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
