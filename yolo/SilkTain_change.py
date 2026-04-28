import os
from ultralytics import YOLO


def train_two_stages():
    # ==========================================
    # 第一阶段：冻结 Backbone，训练检测头 (50 Epochs)
    # ==========================================
    print(">>> 开始第一阶段训练：冻结 Backbone")

    # 加载修改后的模型配置，并加载 nano 预训练权重
    model = YOLO(r'D:\Program Files\student code\ultralytics-8.2.26\ultralytics\Silkworm\yolov8-silkworm.yaml').load('yolov8n.pt')

    results_stage1 = model.train(
        data=r'D:\Program Files\student code\ultralytics-8.2.26\ultralytics\Silkworm\data.yaml',  # 你的数据集配置文件 (请确保路径正确)
        epochs=50,  # 训练50轮
        freeze=10,  # 冻结backbone(前10层)
        lr0=0.01,  # 初始学习率
        cos_lr=True,  # 开启余弦退火
        patience=10,  # 早停机制
        name='stage1_frozen_head',
        exist_ok=True,  # << 重要修复项：覆盖同名文件夹，确保路径始终一致
        batch=16,  # 根据显存调整
        device='0',  # GPU id
        workers=0  # << 解决 Windows 1455 报错
    )

    # ==========================================
    # 第二阶段：解冻全部层，端到端微调 (100 Epochs)
    # ==========================================
    print(">>> 开始第二阶段训练：解冻端到端微调")

    # 动态获取第一阶段的最佳权重路径（规避找错路径的风险）
    best_weight_stage1 = os.path.join('runs', 'detect', 'stage1_frozen_head2', 'weights50', 'best.pt')

    # 安全检查：确认文件真的存在，如果不存在抛出清晰的本地错误，不再去 GitHub 乱下
    if not os.path.exists(best_weight_stage1):
        raise FileNotFoundError(f"找不到第一阶段的权重文件：{best_weight_stage1}")

    # 加载第一阶段的权重
    model_stage2 = YOLO(best_weight_stage1)

    results_stage2 = model_stage2.train(
        data=r'D:\Program Files\student code\ultralytics-8.2.26\ultralytics\Silkworm\data.yaml',
        epochs=100,  # 解冻后训练100轮
        freeze=0,  # 解冻所有层
        lr0=0.01,  # 保持初始学习率配置，继续余弦退火
        cos_lr=True,
        patience=10,
        name='stage2_unfrozen_finetune',
        exist_ok=True,  # 同理，覆盖同名文件夹
        batch=16,
        device='0',
        workers=0
    )


if __name__ == '__main__':
    train_two_stages()
