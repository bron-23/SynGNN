import argparse
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import os

from logger import FileLogger
from datasets.md17 import MD17 # <-- 导入新的数据集类
from nets import model_entrypoint
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer
from engine_md17 import train_one_epoch_md17, evaluate_md17 # <-- 导入新的引擎函数
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils import ModelEmaV2, NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('MD17 Training Script', add_help=False)
    # General
    parser.add_argument('--exp-name', type=str, default='md17_experiment')
    parser.add_argument('--output-dir', type=str, default='outputs/md17_results')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument('--patience-epochs', type=int, default=30,
                        help='Patience for early stopping (default: 50)')
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.9999,
                        help='decay factor for model weights moving average (default: 0.9999)')

    # Dataset
    parser.add_argument('--data-path', type=str, default='data/md17', help='Directory for MD17 .npz files')
    parser.add_argument('--molecule-type', type=str, default='aspirin', help='Molecule to train on')
    parser.add_argument('--max-train-samples', type=int, default=1000)
    parser.add_argument('--max-val-samples', type=int, default=1000)
    parser.add_argument('--max-test-samples', type=int, default=10000)
    parser.add_argument('--delta-frame', type=int, default=1)

    # Model
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2')
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--num-basis', type=int, default=128)
    parser.add_argument('--drop-path', type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-8)
    parser.add_argument('--loss', type=str, default='l1', help='l1 for MAE-like, l2 for MSE')
    # ===================== 在这里添加或补全以下代码 =====================
    # Learning rate schedule parameters (to match timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # 下面这些是cosine调度器可能需要的其他参数，最好也加上
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate')
    # =====================================================================
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    # Workers
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true', default=True)

    # ===================== 在这里添加以下代码 =====================
    # Optimizer Parameters (to match optim_factory.py)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # ==============================================================
    # EMPP self-supervisied learning
    parser.add_argument('--ssp', action='store_true',
                        help='using EMPP as auxiliary tasks')

    # EMPP self-supervised learning
    parser.add_argument('--empp-loss-weight', type=float, default=1.0,  # New or updated
                        help='Weight for the EMPP loss component')
    parser.add_argument('--empp-num-mask', type=int, default=1,  # New
                        help='Number of atoms to mask simultaneously per molecule for EMPP')
    parser.add_argument('--empp-ssp-feature-dim', type=str, default='16x0e',  # New
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

torch.autograd.set_detect_anomaly(True)

def main_for_molecule(args):
    # Setup logger for the specific molecule run
    mol_output_dir = os.path.join(args.output_dir, args.molecule_type)
    Path(mol_output_dir).mkdir(parents=True, exist_ok=True)
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=mol_output_dir)
    args.logger = _log
    _log.info(f"--- Starting training for {args.molecule_type} ---")
    _log.info(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_config = {'train': 10000, 'valid': 2000, 'test': 2000}

    # 如果是 Ethanol 或 Malonaldehyde，就使用特殊的 100k 训练集
    if args.molecule_type in ['ethanol', 'malonaldehyde']:
        _log.info(f"Using special 100k training split for {args.molecule_type}.")
        split_config['train'] = 150000
    # --- Dataset and DataLoader ---
    _log.info("Loading datasets...")
    train_dataset = MD17(args.data_path, args.molecule_type, 'train', max_samples=args.max_train_samples,
                         delta_frame=args.delta_frame,
                         split_sizes=split_config)
    val_dataset = MD17(args.data_path, args.molecule_type, 'valid', max_samples=args.max_val_samples,
                       delta_frame=args.delta_frame,
                         split_sizes=split_config)
    test_dataset = MD17(args.data_path, args.molecule_type, 'test', max_samples=args.max_test_samples,
                        delta_frame=args.delta_frame,
                         split_sizes=split_config)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=args.pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=args.pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=args.pin_mem)
    
    _log.info("--- Dataset Sizes ---")
    _log.info(f"Training set size:   {len(train_dataset):,}") # 使用:,格式化数字，如 100,000
    _log.info(f"Validation set size: {len(val_dataset):,}")
    _log.info(f"Test set size:       {len(test_dataset):,}")
    _log.info("---------------------")

    # --- Model ---
    _log.info("Creating model...")
    create_model = model_entrypoint(args.model_name)
    model = create_model(
        irreps_in=None,
        radius=args.radius,
        num_basis=args.num_basis,
        task_type='md17',  # <-- 关键: 告诉模型这是MD17任务
        drop_path=args.drop_path,
        # 传递所有辅助任务参数
        ssp=args.ssp,
        empp_num_mask=args.empp_num_mask,
        empp_ssp_feature_dim=args.empp_ssp_feature_dim,
        empp_atom_type_embed_irreps=args.empp_atom_type_embed_irreps,
        empp_prioritize_heteroatoms=args.empp_prioritize_heteroatoms,
        empp_pos_pred_num_s2_channels=args.empp_pos_pred_num_s2_channels,
        empp_pos_pred_res_s2grid=args.empp_pos_pred_res_s2grid,
        empp_pos_pred_temp_softmax=args.empp_pos_pred_temp_softmax,
        empp_pos_pred_temp_label=args.empp_pos_pred_temp_label,
        enable_contrastive_learning=args.enable_contrastive,
        contrastive_projection_dim=args.contrastive_projection_dim,
        contrastive_loss_weight=args.contrastive_loss_weight,


    ).to(device)
    _log.info(f"Number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer, Scheduler, Loss ---
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    args.loss = 'l2'  # 确保命令行参数或默认值是l2
    criterion = torch.nn.L1Loss() if args.loss == 'l1' else torch.nn.MSELoss()

    # --- AMP ---
    amp_autocast = suppress if not args.amp else torch.cuda.amp.autocast
    loss_scaler = NativeScaler() if args.amp else None

    # --- Training Loop ---
    best_val_mse = float('inf')
    best_test_mse = float('inf')

    # model_ema = None
    # if args.model_ema:
    #     # 在模型被移动到GPU之后创建EMA对象
    #     model_ema = ModelEmaV2(
    #         model,
    #         decay=args.model_ema_decay,
    #         device=device # 在同一个设备上进行EMA更新，效率更高
    #     )
    #     _log.info(f"Using Model EMA with decay {args.model_ema_decay}")

    start_epoch = 0
    best_val_mse = float('inf')
    best_test_mse_at_best_val = float('inf')
    best_epoch = 0

    if hasattr(args, 'resume') and args.resume:
        if os.path.isfile(args.resume):
            _log.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            try:
                # 使用 strict=False 更稳健
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                
                # ★ 核心：从检查点中恢复 epoch 和 best_val_mse ★
                start_epoch = checkpoint['epoch'] + 1
                best_val_mse = checkpoint['best_val_mse']
                best_epoch = checkpoint['epoch']
                # if model_ema is not None and checkpoint.get('model_ema_state_dict'):
                #     model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
                #     _log.info("Loaded EMA model state from checkpoint.")
                
                _log.info(f"=> loaded checkpoint '{args.resume}' (resuming from epoch {start_epoch})")
                
                # (可选) 恢复 best_test_mse, 如果检查点里存了的话
                if 'best_test_mse' in checkpoint:
                    best_test_mse_at_best_val = checkpoint['best_test_mse']

            except Exception as e:
                _log.error(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
                best_val_mse = float('inf')
        else:
            _log.error(f"=> no checkpoint found at '{args.resume}'. Starting from scratch.")
    else:
        _log.info("=> not resuming from a checkpoint. Starting from scratch.")
    
    # --- Early Stopping 初始化 ---
    patience = args.patience_epochs
    early_stop_counter = 0
     

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()

        train_mse, _ = train_one_epoch_md17(
            model, criterion, train_loader, optimizer, device, epoch, args, 
            loss_scaler=loss_scaler,
            amp_autocast=amp_autocast,
            # model_ema=model_ema # <-- 传入EMA对象
        )

        lr_scheduler.step(epoch)

        model_to_evaluate =  model
        
        val_mse = evaluate_md17(
            model=model_to_evaluate, # <-- 使用EMA模型
            data_loader=val_loader, 
            device=device,
            args=args, 
            amp_autocast=amp_autocast
        )

        # 在验证集上评估

        is_best = val_mse < best_val_mse
        # 检查是否是当前最佳的验证集结果
        if is_best:
            # 更新最佳验证集MSE
            best_val_mse = val_mse
            best_epoch = epoch
            # ★ 关键：在找到新的最佳验证模型时，立即在测试集上评估并记录对应的测试集MSE
            best_test_mse_at_best_val = evaluate_md17(
                model=model_to_evaluate,
                data_loader=test_loader, device=device,
                args=args, amp_autocast=amp_autocast,
            )

            _log.info(
                f"*** New Best Val MSE (x10^-2): {best_val_mse * 100:.4f}, "
                f"Corresponding Test MSE (x10^-2): {best_test_mse_at_best_val * 100:.4f} at Epoch {epoch} ***"
            )
            # 重置计数器，因为我们找到了一个更好的模型
            early_stop_counter = 0
            # ==================== ★★★ 保存最佳模型 ★★★ ====================
            # 我们只保存性能最好的模型
            best_checkpoint_path = os.path.join(mol_output_dir, 'best_model.pth')
            _log.info(f"Saving best model to {best_checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_mse': best_val_mse,
                'best_test_mse': best_test_mse_at_best_val, # 保存测试结果
                'args': args,
                # 'model_ema_state_dict': model_ema.module.state_dict() if model_ema is not None else None # <-- 保存EMA状态
            }, best_checkpoint_path)
            # ===================================================================
        else:
            # 验证集性能没有提升，增加计数器
            early_stop_counter += 1
            _log.info(f"Validation MSE did not improve. Early stopping counter: {early_stop_counter}/{patience}")

        # ==================== ★★★ 早停判断 ★★★ ====================
        if early_stop_counter >= patience:
            _log.info(f"Validation performance has not improved for {patience} epochs. Triggering early stopping.")
            break # ★ 跳出 for 循环，提前结束训练 ★
        # =============================================================

        # ==================== ★★★ (可选) 保存最新模型 ★★★ ====================
        # 每隔一定epoch保存一次，用于意外中断后的恢复
        if (epoch + 1) % 100 == 0: # 例如每100个epoch保存一次
            latest_checkpoint_path = os.path.join(mol_output_dir, 'latest_checkpoint.pth')
            _log.info(f"Saving latest checkpoint to {latest_checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_mse': best_val_mse,
                'args': args,
            }, latest_checkpoint_path)
        # ===================================================================

        # 打印每个epoch的总结日志
        _log.info(
            f"Epoch {epoch} Summary | Train MSE (x10^-2): {train_mse * 100:.4f} | "
            f"Val MSE (x10^-2): {val_mse * 100:.4f} | Time: {time.perf_counter() - epoch_start_time:.2f}s"
        )

        # --- 训练循环结束后 ---
    _log.info(f"--- Finished training for {args.molecule_type} ---")
    final_test_mse = best_test_mse_at_best_val
    _log.info(f"Final Reported Test MSE (from best EMA model during training, x10^-2): {final_test_mse * 100 if final_test_mse != float('inf') else 'N/A'}")


    return best_val_mse, final_test_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MD17 Main Runner', parents=[get_args_parser()])
    args = parser.parse_args()

    # molecules = [
    #     'aspirin', 'benzene', 'ethanol', 'malonaldehyde',
    #     'naphthalene', 'salicylic', 'toluene', 'uracil'
    # ]
    molecules = [
        'malonaldehyde','ethanol'
    ]

    results = {}

    for mol in molecules:
        args.molecule_type = mol
        # Optional: Adjust hyperparameters per molecule if needed
        # if mol == 'aspirin': args.lr = ...
        val_mae, test_mae = main_for_molecule(args)
        results[mol] = {'val_mae': val_mae, 'test_mae': test_mae}

    print("\n\n" + "=" * 50)
    print("           MD17 FINAL RESULTS SUMMARY (Prediction error (x10^-2))")
    print("=" * 50)
    # 修改表头
    print(f"{'Molecule':<20} | {'Test MSE (x10^-2)':<25}")
    print("-" * 50)

    all_test_mses = []
    for mol, res in results.items():
        # ★★★ 关键：在打印最终表格时进行缩放 ★★★
        print(f"{mol:<20} | {res['test_mae'] * 100:.4f}")
        all_test_mses.append(res['test_mae'])

    if all_test_mses:
        avg_test_mse = np.mean(all_test_mses)
        print("-" * 50)
        # ★★★ 关键：在打印平均值时进行缩放 ★★★
        print(f"{'Average':<20} | {avg_test_mse * 100:.4f}")
    print("=" * 50)

    # Save final results to a json file
    results['average_test_mae'] = avg_test_mse

    final_results_path = os.path.join(args.output_dir, f'{args.exp_name}_summary.json')
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Final summary saved to {final_results_path}")