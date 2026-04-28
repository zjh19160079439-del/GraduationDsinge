import os
import numpy as np
from ultralytics import YOLO


def evaluate_best_model():
    print(">>> 正在初始化评估程序...")

    # ==========================================
    # 1. 请在这里修改为你的实际路径
    # ==========================================
    # 你第二阶段训练得到的最佳权重路径
    weights_path = r'D:\Program Files\student code\ultralytics-8.2.26\runs\detect\SilkTrain01\weights\best.pt'
    # 你的数据集配置文件路径
    data_yaml_path = r'D:\Program Files\student code\ultralytics-8.2.26\ultralytics\Silkworm\data.yaml'

    # 安全检查
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件，请检查路径: {weights_path}")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"找不到数据集配置，请检查路径: {data_yaml_path}")

    # ==========================================
    # 2. 加载模型并执行验证
    # ==========================================
    print(f">>> 正在加载最佳模型: {weights_path}")
    model = YOLO(weights_path)

    print(">>> 正在验证集上进行推理与测速，请耐心等待（这可能需要一两分钟）...")
    # 运行验证（设置 split='val' 确保在验证集上评估）
    metrics = model.val(data=data_yaml_path, split='val', verbose=False)

    # ==========================================
    # 3. 提取精度指标
    # ==========================================
    # YOLOv8 的 metrics 对象包含一个 results_dict 字典，里面存有具体数值
    precision = metrics.results_dict.get('metrics/precision(B)', 0)
    recall = metrics.results_dict.get('metrics/recall(B)', 0)
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)

    # 提取最佳 F1-Score
    # metrics.box.f1 是一个数组，记录了不同置信度下的 F1 分数。我们取它的最大值。
    f1_curve = metrics.box.f1
    if f1_curve is not None and len(f1_curve) > 0:
        # 如果是单类别（只有小蚕），直接取最大值；多类别则取平均后的最大值
        best_f1 = np.max(np.mean(f1_curve, axis=0)) if f1_curve.ndim > 1 else np.max(f1_curve)
    else:
        best_f1 = 0.0

    # ==========================================
    # 4. 提取速度指标
    # ==========================================
    speeds = metrics.speed
    # 总处理时间 = 预处理时间 + 推理时间 + 后处理时间 (单位: 毫秒)
    avg_time_ms = speeds.get('preprocess', 0) + speeds.get('inference', 0) + speeds.get('postprocess', 0)

    # 计算 FPS (1秒 = 1000毫秒)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0

    # ==========================================
    # 5. 格式化输出结果
    # ==========================================
    print("\n" + "=" * 45)
    print("🏆 小蚕识别模型 (best.pt) 最终性能评估报告 🏆")
    print("=" * 45)
    print(f"🔹 Precision (精确率) : {precision * 100:.2f} %")
    print(f"🔹 Recall    (召回率) : {recall * 100:.2f} %")
    print(f"🔹 mAP@0.5   (平均精度): {map50 * 100:.2f} %")
    print(f"🔹 F1-Score  (最高得分): {best_f1:.4f}")
    print("-" * 45)
    print(f"⏱️ 平均处理时间 (每张图): {avg_time_ms:.2f} ms")
    print(f"🚀 综合处理帧率 (FPS)  : {fps:.2f} 帧/秒")
    print("=" * 45)
    print("注：速度指标受当前运行硬件(CPU/GPU)影响。如需测试部署速度，请在最终部署设备上运行此脚本。")


if __name__ == "__main__":
    evaluate_best_model()