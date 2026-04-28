# images_dir = r"D:\GraduationDsign\datasets\yolo_dataset\images\train"
# labels_dir = r"D:\GraduationDsign\datasets\yolo_dataset\labels\train"
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob



def calculate_iou(boxA, boxB):
    """计算交并比 (IoU), box 格式为 [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0: return 0.0
    return interArea / unionArea


def evaluate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """计算评测指标"""
    TP = 0;
    FP = 0;
    FN = 0
    matched_gt_indices = set()
    for p_box in pred_boxes:
        best_iou = 0;
        best_gt_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt_indices: continue
            iou = calculate_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou;
                best_gt_idx = i
        if best_iou >= iou_threshold:
            TP += 1;
            matched_gt_indices.add(best_gt_idx)
        else:
            FP += 1
    FN = len(gt_boxes) - len(matched_gt_indices)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    map_05 = precision * recall
    return precision, recall, f1_score, map_05


# --- 读取真实的 YOLO 格式标注数据 ---
def load_yolo_annotations(txt_path, img_width, img_height):
    gt_boxes = []
    if not os.path.exists(txt_path):
        return gt_boxes

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # 解析归一化坐标
                class_id, x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[:5])
                # 还原为绝对像素值
                abs_w = int(w_norm * img_width)
                abs_h = int(h_norm * img_height)
                abs_x_center = x_center_norm * img_width
                abs_y_center = y_center_norm * img_height
                # 转换为左上角坐标 (x, y)
                x = int(abs_x_center - (abs_w / 2))
                y = int(abs_y_center - (abs_h / 2))
                gt_boxes.append([x, y, abs_w, abs_h])

    return gt_boxes


# --- 核心图像处理流水线 ---
def process_image(image_path):
    start_time = time.time()
    img = cv2.imread(image_path)
    if img is None: return [], img, 0, 0
    img_h, img_w = img.shape[:2]

    # 1. 颜色分割 2. 去噪 3. 分水岭
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 6]), np.array([108, 82, 247]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    ret, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1;
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    predicted_boxes = []
    for label in range(2, ret + 1):
        target_mask = np.zeros(mask.shape, dtype="uint8")
        target_mask[markers == label] = 255
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 300 < cv2.contourArea(cnt) < 15000:
                predicted_boxes.append(list(cv2.boundingRect(cnt)))

    p_time = time.time() - start_time
    fps = 1.0 / p_time if p_time > 0 else 0
    return predicted_boxes, img, p_time, fps



def draw_yolo_style_box(img, box, color, label):
    """
    绘制类似 YOLO 风格的边界框，包含带有背景色的文字标签
    """
    x, y, w, h = box
    # 1. 画边界框
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # 2. 准备文字和字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # 获取文字大小以绘制背景矩形
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # 防止标签背景画到图片外面（如果框太靠近顶部）
    top_y = y - text_height - baseline - 2
    if top_y < 0:
        # 画在框内部
        cv2.rectangle(img, (x, y), (x + text_width, y + text_height + baseline + 2), color, -1)
        cv2.putText(img, label, (x, y + text_height + baseline - 1), font, font_scale, (255, 255, 255), thickness)
    else:
        # 画在框外部上方
        cv2.rectangle(img, (x, top_y), (x + text_width, y), color, -1)
        cv2.putText(img, label, (x, y - baseline - 1), font, font_scale, (255, 255, 255), thickness)



def plot_yolo_style_charts(tp, fp, fn, save_dir, precision, recall, f1):
    """生成类似 YOLO 的混淆矩阵和 PR/F1 曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 混淆矩阵 (Confusion Matrix)
    # y轴: True (Silkworm, Background), x轴: Predicted (Silkworm, Background)
    matrix = np.array([
        [tp, fn],  # True 目标被预测为目标(TP)，被预测为背景(FN)
        [fp, 0]  # True 背景被预测为目标(FP)，预测为背景(传统CV无负样本边界框，TN设为0)
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=["Silkworm", "Background"],
                yticklabels=["Silkworm", "Background"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 归一化混淆矩阵
    row_sums = matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # 防止除0
    matrix_norm = matrix.astype('float') / row_sums
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Silkworm", "Background"],
                yticklabels=["Silkworm", "Background"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Normalized")
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 伪 PR 曲线 (Precision-Recall Curve)
    # 传统算法无“置信度”区间，因此曲线由原点至当前点构成一个阶梯矩形
    plt.figure(figsize=(6, 5))
    plt.plot([0, recall, recall], [precision, precision, 0], 'b-', linewidth=2,
             label=f'all classes {precision:.2f} mAP@0.5')
    plt.fill_between([0, recall], [precision, precision], alpha=0.1, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, "PR_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 伪 F1 曲线 (F1-Confidence Curve)
    plt.figure(figsize=(6, 5))
    # 假设置信度在 [0, 1.0] 时 F1 保持恒定
    plt.plot([0, 1.0], [f1, f1], 'b-', linewidth=2, label=f'all classes {f1:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Confidence")
    plt.ylabel("F1")
    plt.title("F1-Confidence Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, "F1_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_dataset(images_dir, labels_dir, save_dir="runs/val"):
    """批量评估整个数据集"""
    os.makedirs(save_dir, exist_ok=True)
    # 支持抓取所有的 .jpg 或 .png 格式图片
    image_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(os.path.join(images_dir, "*.png"))

    if len(image_paths) == 0:
        print(f"⚠ 警告: 在 {images_dir} 中未找到任何图片。")
        return

    global_TP = 0;
    global_FP = 0;
    global_FN = 0
    total_time = 0

    print(f" 开始批量评估，共找到 {len(image_paths)} 张图片...")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")

        # 预测单张图片
        preds, img, p_time, _ = process_image(img_path)
        total_time += p_time

        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 读取 GT
        gts = load_yolo_annotations(txt_path, img_w, img_h)

        # 累计此图的 TP, FP, FN
        matched_gt_indices = set()
        for p_box in preds:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gts):
                if i in matched_gt_indices: continue
                iou = calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou >= 0.5:
                global_TP += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                global_FP += 1
        global_FN += (len(gts) - len(matched_gt_indices))

    # ==========================================
    # 汇总计算全局大盘指标 (新增/完善各项指标)
    # ==========================================
    precision = global_TP / (global_TP + global_FP) if (global_TP + global_FP) > 0 else 0.0
    recall = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    map_05 = precision * recall  # 传统CV的mAP@0.5近似计算(即PR曲线下的面积)

    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0


    # 2. 将结果写入文本文件 (模仿 YOLO 的 results.txt)
    results_txt_path = os.path.join(save_dir, "results.txt")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n")
        f.write(f"Images Processed : {len(image_paths)}\n")
        f.write(f"TP / FP / FN     : {global_TP} / {global_FP} / {global_FN}\n")
        f.write("-" * 18 + "\n")
        f.write(f"Precision        : {precision:.4f}\n")
        f.write(f"Recall           : {recall:.4f}\n")
        f.write(f"F1-Score         : {f1_score:.4f}\n")
        f.write(f"mAP@0.5          : {map_05:.4f}\n")
        f.write("-" * 18 + "\n")
        f.write(f"Avg Time per img : {avg_time * 1000:.1f} ms\n")
        f.write(f"FPS              : {fps:.1f}\n")

    # 3. 绘制并保存 YOLO 风格图表
    plot_yolo_style_charts(global_TP, global_FP, global_FN, save_dir, precision, recall, f1_score)
    print(f" 所有评估结果和图表已保存至: {os.path.abspath(save_dir)}")
    print(f" 查看详细数据请打开: {os.path.abspath(results_txt_path)}")


def generate_dummy_data_if_missing(img_path, txt_path):
    """辅助功能：如果没有找到测试数据，则自动生成一张带有假标签的图像进行演示"""
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(" 未找到指定的图像或标签文件，正在生成虚拟数据用于测试演示...")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        # 创建暗绿色背景 (模拟桑叶) 和一个亮色块 (模拟小蚕)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = [50, 100, 50]
        cv2.circle(img, (200, 200), 20, (180, 230, 240), -1)
        cv2.imwrite(img_path, img)

        # 写入虚拟的 YOLO 格式标签 (class, x_c, y_c, w, h)
        with open(txt_path, 'w') as f:
            f.write("0 0.50 0.50 0.10 0.10\n")


if __name__ == "__main__":
    images_dir =  r"D:\GraduationDsign\datasets\yolo_dataset\images\train"
    labels_dir = r"D:\GraduationDsign\datasets\yolo_dataset\labels\train"


    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        generate_dummy_data_if_missing(os.path.join(images_dir, "0000_aug_0.jpg"),
                                       os.path.join(labels_dir, "0000_aug_0.txt"))
        generate_dummy_data_if_missing(os.path.join(images_dir, "0001_aug_0.jpg"),
                                       os.path.join(labels_dir, "0001_aug_0.txt"))

    # 执行全图批量评估，并将生成的图表统一存放到 runs/val 文件夹 (高度模仿YOLO的默认结构)
    evaluate_dataset(images_dir, labels_dir, save_dir="runs/val")