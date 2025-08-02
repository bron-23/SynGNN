# engine.py
import torch
import torch.nn.functional as F
from typing import Iterable, Optional, Dict
from timm.utils import ModelEmaV2, dispatch_clip_grad
import time
import math  # For math.ceil in augmentation

# Assuming torch_geometric is used for graph data and pooling
try:
    import torch_geometric.data
    import torch_geometric.nn as pyg_nn  # For Batch.from_data_list and potential global_pool
except ImportError:
    print("torch_geometric not found. Please install it if you are working with graph data.")
    torch_geometric = None  # Graceful fallback or raise error

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("tensorboard not found. Logging to TensorBoard will be disabled.")
    SummaryWriter = None


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Helper function for contrastive data augmentation
def create_feature_masked_view_for_contrastive(
        graph_data: torch_geometric.data.Data,
        mask_ratio: float,
        mask_token_idx: int,
        # 新增的参数
        prioritize_heteroatoms: bool,
        heteroatom_indices_tensor: torch.Tensor,
        atom_map_tensor: torch.Tensor
) -> torch_geometric.data.Data:
    augmented_graph = graph_data.clone()
    num_nodes = augmented_graph.num_nodes
    # 如果不增强或图为空，直接返回
    if not prioritize_heteroatoms or num_nodes == 0 or mask_ratio <= 0.0:
        # 如果不优先处理杂原子，就执行原始的随机掩码逻辑
        # (为了代码清晰，这里可以把之前的随机掩码逻辑放在这里作为else分支)
        # ... 此处省略原始的随机掩码代码 ...
        perm = torch.randperm(num_nodes, device=graph_data.z.device)
        num_to_mask = math.ceil(num_nodes * mask_ratio)
        masked_node_indices = perm[:num_to_mask]
        if masked_node_indices.numel() > 0:
            new_z = augmented_graph.z.clone()
            new_z[masked_node_indices] = mask_token_idx
            augmented_graph.z = new_z
        return augmented_graph

    # --- 开始执行杂原子优先策略 ---

    # 1. 计算需要掩码的总数
    num_to_mask = math.ceil(num_nodes * mask_ratio)
    if num_to_mask >= num_nodes: num_to_mask = num_nodes - 1
    if num_to_mask == 0: return augmented_graph

    # 2. 将原始原子序数(Z值)映射到内部索引
    raw_z = augmented_graph.z
    # 确保张量在同一设备上
    device = raw_z.device
    atom_map_tensor = atom_map_tensor.to(device)
    heteroatom_indices_tensor = heteroatom_indices_tensor.to(device)

    clamped_z = torch.clamp(raw_z, 0, len(atom_map_tensor) - 1)
    internal_node_indices = atom_map_tensor[clamped_z]

    all_graph_indices = torch.arange(num_nodes, device=device)

    # 3. 区分杂原子和非杂原子
    is_hetero_mask = torch.isin(internal_node_indices, heteroatom_indices_tensor)

    hetero_nodes = all_graph_indices[is_hetero_mask]
    non_hetero_nodes = all_graph_indices[~is_hetero_mask]

    masked_node_indices_final = []

    # 4. 优先从杂原子中随机采样
    num_from_hetero = min(num_to_mask, len(hetero_nodes))
    if num_from_hetero > 0:
        perm = torch.randperm(len(hetero_nodes), device=device)
        selected_hetero_indices = hetero_nodes[perm[:num_from_hetero]]
        masked_node_indices_final.append(selected_hetero_indices)

    # 5. 如果数量不足，再从非杂原子中随机采样
    remaining_to_mask = num_to_mask - num_from_hetero
    if remaining_to_mask > 0 and len(non_hetero_nodes) > 0:
        num_from_non_hetero = min(remaining_to_mask, len(non_hetero_nodes))
        perm = torch.randperm(len(non_hetero_nodes), device=device)
        selected_non_hetero_indices = non_hetero_nodes[perm[:num_from_non_hetero]]
        masked_node_indices_final.append(selected_non_hetero_indices)

    # 6. 合并并执行掩码
    if masked_node_indices_final:
        final_indices_to_mask = torch.cat(masked_node_indices_final)
        new_z = augmented_graph.z.clone()
        new_z[final_indices_to_mask] = mask_token_idx
        augmented_graph.z = new_z

    return augmented_graph


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list,
                    target_idx_in_y: int,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args,  # 传入所有命令行参数，方便访问
                    model_ema: Optional[ModelEmaV2] = None,
                    amp_autocast=None,
                    loss_scaler=None
                    ):
    model.train()

    # 从args中解包参数，避免函数签名过长
    print_freq = args.print_freq
    ssp_active = args.ssp
    empp_loss_weight = args.empp_loss_weight
    contrastive_active = args.enable_contrastive
    contrastive_loss_weight = args.contrastive_loss_weight
    contrastive_temp = args.contrastive_temp
    contrastive_aug_mask_ratio = args.contrastive_aug_mask_ratio
    contrastive_mask_token_idx = args.contrastive_mask_token_idx
    clip_grad = args.clip_grad
    logger = args.logger  # 假设logger也被添加到args中

    loss_metric_total = AverageMeter()
    loss_metric_main = AverageMeter()
    loss_metric_empp = AverageMeter()
    loss_metric_contrastive = AverageMeter()
    mae_metric_main = AverageMeter()

    start_time_epoch = time.perf_counter()
    task_mean, task_std = norm_factor[0], norm_factor[1]

    # 从模型中获取原子映射信息
    model_to_call = model.module if hasattr(model, 'module') else model
    use_prio_for_contrastive = getattr(args, 'contrastive_prioritize_heteroatoms', False)
    hetero_indices_tensor = model_to_call.heteroatom_internal_indices
    atom_map_tensor = model_to_call.internal_atom_map_tensor

    for step, data_orig_batch in enumerate(data_loader):
        step_start_time = time.perf_counter()

        # 1. 数据准备
        data_orig_batch = data_orig_batch.to(device, non_blocking=True)
        batch_size = data_orig_batch.num_graphs

        try:
            main_task_target_values = data_orig_batch.y[:, target_idx_in_y]
        except (IndexError, TypeError):
            main_task_target_values = data_orig_batch.y
        main_task_target_normalized = (main_task_target_values - task_mean) / task_std

        augmented_data_tuple = None
        if contrastive_active:
            if torch_geometric is None:
                raise ImportError("torch_geometric is required for contrastive learning augmentation.")

            augmented_data_list = [
                create_feature_masked_view_for_contrastive(
                    d,  # 直接传递，函数内部会clone
                    mask_ratio=contrastive_aug_mask_ratio,
                    mask_token_idx=contrastive_mask_token_idx,
                    prioritize_heteroatoms=use_prio_for_contrastive,
                    heteroatom_indices_tensor=hetero_indices_tensor,
                    atom_map_tensor=atom_map_tensor
                ) for d in data_orig_batch.to_data_list()
            ]
            if augmented_data_list:
                positive_batch = torch_geometric.data.Batch.from_data_list(augmented_data_list).to(device)
                augmented_data_tuple = (positive_batch.pos, positive_batch.batch, positive_batch.z, positive_batch.ptr)

        # 2. 模型前向传播 (单次调用)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast() if amp_autocast is not None else torch.no_grad():
            model_outputs = model(
                pos=data_orig_batch.pos, batch=data_orig_batch.batch,
                node_atom=data_orig_batch.z, ptr=data_orig_batch.ptr,
                main_task_target_for_loss_and_ssp_injection=main_task_target_normalized,
                ssp_active_flag=ssp_active,
                contrastive_active_flag=contrastive_active,
                contrastive_augmented_data=augmented_data_tuple
            )

            # 3. 损失计算与聚合
            main_pred = model_outputs.get('main_pred')
            main_loss = criterion(main_pred.squeeze(-1) if main_pred.ndim > 1 else main_pred,
                                  main_task_target_normalized)
            total_loss = main_loss
            loss_metric_main.update(main_loss.item(), n=batch_size)

            if ssp_active and 'empp_loss' in model_outputs:
                empp_loss = model_outputs.get('empp_loss', torch.tensor(0.0, device=device))
                if not torch.isnan(empp_loss) and not torch.isinf(empp_loss):
                    total_loss += empp_loss_weight * empp_loss
                    loss_metric_empp.update(empp_loss.item(), n=batch_size)

            if contrastive_active and 'z_original_contrastive' in model_outputs and 'z_augmented_contrastive' in model_outputs:
                z_original = model_outputs['z_original_contrastive']
                z_positive = model_outputs['z_augmented_contrastive']
                # InfoNCE Loss 的计算
                logits = torch.matmul(z_original, z_positive.T) / contrastive_temp
                labels = torch.arange(z_original.shape[0], device=device)
                cl_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
                if not torch.isnan(cl_loss) and not torch.isinf(cl_loss):
                    total_loss += contrastive_loss_weight * cl_loss
                    loss_metric_contrastive.update(cl_loss.item(), n=batch_size)

        # 4. 反向传播与优化
        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())
        else:
            total_loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        # 5. 日志记录
        pred_denorm = main_pred.detach() * task_std + task_mean
        mae = F.l1_loss(pred_denorm.squeeze(), main_task_target_values)
        mae_metric_main.update(mae.item(), n=batch_size)
        loss_metric_total.update(total_loss.item(), n=batch_size)

        if logger and (step % print_freq == 0 or step == len(data_loader) - 1):
            log_items = [f"Epoch: [{epoch}][{step}/{len(data_loader)}]", f"Loss: {loss_metric_total.avg:.5f}"]
            if ssp_active: log_items.append(f"EMPP_Raw: {loss_metric_empp.avg:.5f}")
            if contrastive_active: log_items.append(f"Contra_Raw: {loss_metric_contrastive.avg:.5f}")
            log_items.extend([
                f"MAE_main: {mae_metric_main.avg:.5f}",
                f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                f"Time/step: {(time.perf_counter() - step_start_time) * 1000:.0f}ms"
            ])
            logger.info('\t'.join(log_items))

    if logger:
        logger.info(
            f"Epoch {epoch} Training Summary: Avg Total Loss: {loss_metric_total.avg:.5f}, Avg Main MAE: {mae_metric_main.avg:.5f}, Time: {time.perf_counter() - start_time_epoch:.2f}s"
        )

    return mae_metric_main.avg


def evaluate(model: torch.nn.Module, norm_factor: list, target_idx_in_y: int,
             data_loader: Iterable, device: torch.device,
             amp_autocast=None, print_freq: int = 100, logger=None,
             tensorboard_writer: Optional[SummaryWriter] = None,
             epoch_num: Optional[int] = None, log_prefix: str = 'Eval'
             ):
    model.eval()
    # For evaluation, we only care about the main task's performance
    # Criterion for main task (e.g., L1 for MAE)
    criterion = torch.nn.L1Loss()  # Or MSELoss if that's your main metric for loss display

    loss_metric_main_eval = AverageMeter()  # Tracks main task loss on eval set
    mae_metric_main_eval = AverageMeter()  # Tracks main task MAE on eval set

    task_mean, task_std = norm_factor[0], norm_factor[1]

    with torch.no_grad():
        for step, data_eval_batch in enumerate(data_loader):
            data_eval_batch = data_eval_batch.to(device, non_blocking=True)

            if data_eval_batch.y.ndim > 1 and data_eval_batch.y.shape[1] > target_idx_in_y:
                main_task_target_values_eval = data_eval_batch.y[:, target_idx_in_y]
            # ... (handle other cases for main_task_target_values_eval) ...
            else:  # Fallback
                main_task_target_values_eval = data_eval_batch.y
            main_task_target_normalized_eval = (main_task_target_values_eval - task_mean) / task_std

            with amp_autocast() if amp_autocast is not None else torch.no_grad():
                # For evaluation, call model in non-SSP, non-contrastive mode to get main prediction
                # This assumes model.forward correctly handles ssp_active_flag=False and
                # contrastive_active_flag=False to give only main_pred.
                model_outputs_eval = model(
                    pos=data_eval_batch.pos, batch=data_eval_batch.batch,
                    node_atom=data_eval_batch.z, ptr=data_eval_batch.ptr,
                    main_task_target_for_loss_and_ssp_injection=None,  # No injection
                    ssp_active_flag=False,
                    contrastive_active_flag=False
                )
                main_pred_eval = model_outputs_eval['main_pred']  # Assuming forward returns dict
                main_pred_eval = main_pred_eval.squeeze(-1) if main_pred_eval.ndim > 1 else main_pred_eval

            current_eval_loss = criterion(main_pred_eval, main_task_target_normalized_eval)
            loss_metric_main_eval.update(current_eval_loss.item(), main_pred_eval.shape[0])

            main_pred_eval_original_scale = main_pred_eval.detach() * task_std + task_mean
            err_main_task_eval = main_pred_eval_original_scale - main_task_target_values_eval
            mae_main_task_eval_current_step = torch.mean(torch.abs(err_main_task_eval))
            mae_metric_main_eval.update(mae_main_task_eval_current_step.item(), main_pred_eval.shape[0])

            if logger is not None and print_freq > 0 and (step % print_freq == 0 or step == len(data_loader) - 1):
                logger.info(
                    f'{log_prefix} Step [{step}/{len(data_loader)}]: Current MAE: {mae_main_task_eval_current_step.item():.5f}, Avg MAE: {mae_metric_main_eval.avg:.5f}')

    if tensorboard_writer is not None and epoch_num is not None:
        tensorboard_writer.add_scalar(f'Epoch_Loss/{log_prefix}_MainTask_Avg', loss_metric_main_eval.avg, epoch_num)
        tensorboard_writer.add_scalar(f'Epoch_MAE/{log_prefix}_MainTask_Avg', mae_metric_main_eval.avg, epoch_num)

    if logger is not None:
        logger.info(f'{log_prefix} Summary - Epoch [{epoch_num if epoch_num is not None else "N/A"}]: '
                    f'Avg MainTask Loss: {loss_metric_main_eval.avg:.5f}, Avg MainTask MAE: {mae_metric_main_eval.avg:.5f}')

    return mae_metric_main_eval.avg, loss_metric_main_eval.avg


def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)