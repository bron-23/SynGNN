#!/bin/bash
# =========================================================================
# === Final Recommended Run Script for MD17 Dynamics Prediction Task ===
# =========================================================================
GPU_ID=${1:-0}
BATCH_SIZE=${2:-64} # 从命令行获取，默认为 8
LEARNING_RATE=${3:-2e-4}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
# --- Configuration ---
# 实验名称，方便区分每次运行
EXP_NAME="md17_final_run_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/${EXP_NAME}"
# 硬件配置
DEVICE="cuda:0" # 确保这是一个可用的GPU索引
# 数据集路径
DATA_PATH="data/md17"
# 模型名称
MODEL_NAME="graph_attention_transformer_nonlinear_l2"
# ----------------- ★★★ 合理的训练超参数 ★★★ -----------------
# 1. 训练周期: 对于MD17，500-1000个epoch是比较常见的
EPOCHS=500
# 2. 批次大小: 对于复杂的分子，8或16是常见的大小
# 3. 学习率: 1e-4 到 5e-5 是一个很好的起始范围
# 4. 梯度裁剪: 保持开启，1.0是非常标准且安全的值
CLIP_GRAD=1.0
# 5. 权重衰减: 一个较小的正则化值
WEIGHT_DECAY=1e-4
# 6. 数据集划分:
MAX_TRAIN_SAMPLES=150000
MAX_VAL_SAMPLES=2000
MAX_TEST_SAMPLES=2000 # 设置一个足够大的数来包含所有剩余的测试样本
# 7. 时间步长: 遵循HEGNN论文的设置
DELTA_FRAME=3000
RADIUS=5.0
# 8. 辅助任务:
# 先从只开启EMPP(SSP)开始，这是与主任务最相关的辅助任务。
# 如果训练稳定且效果好，再考虑加入对比学习。
AUX_FLAGS="--ssp --enable-contrastive"
EMPP_LOSS_WEIGHT=0.1
CONTRASTIVE_LOSS_WEIGHT=0.1
 # 辅助任务的权重通常从一个较小的值开始，如0.1
# 9. 混合精度: 先保持禁用，确保稳定。成功跑完一次后，可以尝试去掉 --no-amp 来加速。
AMP_FLAG="--no-amp"
# ---------------------------------------------------------------------
echo "==========================================================="
echo "Starting MD17 Evaluation Run: ${EXP_NAME}"
echo "Using a set of reasonable hyperparameters for training."
echo "==========================================================="
echo "BATCH_SIZE:       ${BATCH_SIZE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "GPU ID:            ${GPU_ID}"
echo "Learning Rate: ${LEARNING_RATE}, Epochs: ${EPOCHS}"
echo "Auxiliary Tasks Enabled: ${AUX_FLAGS:-'None'}"
echo "-----------------------------------------------------------"
# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
# 运行主程序
python main_md17.py \
    --exp-name "${EXP_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --seed 42 \
    --print-freq 50 \
    --data-path "${DATA_PATH}" \
    --delta-frame ${DELTA_FRAME} \
    --model-name "${MODEL_NAME}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --loss 'l2' \
    --sched 'cosine' \
    --warmup-epochs 10 \
    --min-lr 1e-6 \
    --clip-grad ${CLIP_GRAD} \
    --max-train-samples ${MAX_TRAIN_SAMPLES} \
    --max-val-samples ${MAX_VAL_SAMPLES} \
    --max-test-samples ${MAX_TEST_SAMPLES} \
    --workers 8 \
    --radius ${RADIUS} \
    --pin-mem \
    ${AUX_FLAGS}
    --contrastive-loss-weight ${CONTRASTIVE_LOSS_WEIGHT}\
    --empp-loss-weight ${EMPP_LOSS_WEIGHT}

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "==========================================================="
    echo "MD17 evaluation finished successfully."
    echo "Final summary report is located at: ${OUTPUT_DIR}/${EXP_NAME}_summary.json"
    echo "==========================================================="
else
    echo "==========================================================="
    echo "MD17 evaluation failed with an error. Please check the logs."
    echo "==========================================================="
fi