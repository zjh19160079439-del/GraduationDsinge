import os
import json
import cv2
import numpy as np
import albumentations as A
import copy

# ================= 配置区域 =================
INPUT_DIR = "D:\Adatasets\images\one23"  # 原始图片和json所在的文件夹路径 (请修改为你的实际路径)
OUTPUT_DIR = "yolo_datasets/"  # 增强后保存的文件夹路径
AUG_MULTIPLIER = 3  # 每张原始图片生成几张增强图片

# 定义 Albumentations 数据增强流水线
# 这里组合了翻转、旋转、光照、模糊等推荐方法
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 50%概率水平翻转
    A.VerticalFlip(p=0.5),  # 50%概率垂直翻转
    A.RandomRotate90(p=0.5),  # 50%概率随机旋转90度
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),  # 平移、缩放、任意角度旋转
    A.RandomBrightnessContrast(p=0.5),  # 随机亮度和对比度
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # 20%概率高斯模糊
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3)  # 色彩微调
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))


# pascal_voc 格式是指 [x_min, y_min, x_max, y_max]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_labelme_json(json_path):
    """读取 LabelMe 格式的 JSON，提取边界框和标签"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bboxes = []
    category_ids = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            # LabelMe 矩形点通常是 [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 确保坐标顺序为 x_min, y_min, x_max, y_max
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            bboxes.append([x_min, y_min, x_max, y_max])
            category_ids.append(shape['label'])

    return data, bboxes, category_ids


def main():
    ensure_dir(OUTPUT_DIR)

    # 获取所有图片文件
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"找到 {len(files)} 张图片，开始数据增强...")

    for filename in files:
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(INPUT_DIR, filename)
        json_path = os.path.join(INPUT_DIR, base_name + '.json')

        # 检查对应的 json 是否存在
        if not os.path.exists(json_path):
            print(f"警告: 找不到 {filename} 对应的 json 文件，跳过。")
            continue

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"错误: 无法读取图片 {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations 使用 RGB

        # 读取标注
        try:
            original_data, bboxes, category_ids = process_labelme_json(json_path)
        except Exception as e:
            print(f"解析 {json_path} 时出错: {e}")
            continue

        # 如果没有标注框，依然可以增强图像，但为了安全起见，跳过或原样复制
        if not bboxes:
            continue

        # 生成增强数据
        for i in range(AUG_MULTIPLIER):
            try:
                # 应用变换
                transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_category_ids = transformed['category_ids']

                # 新的文件名
                new_base_name = f"{base_name}_aug_{i}"
                new_img_filename = new_base_name + ".jpg"
                new_json_filename = new_base_name + ".json"

                new_img_path = os.path.join(OUTPUT_DIR, new_img_filename)
                new_json_path = os.path.join(OUTPUT_DIR, new_json_filename)

                # 保存增强后的图片
                cv2.imwrite(new_img_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                # 组装新的 JSON 数据
                new_data = copy.deepcopy(original_data)
                new_data['imagePath'] = new_img_filename
                new_data['imageData'] = None  # 清空 base64 图像数据以节省空间，LabelMe读取时会自动加载对应的本地图片
                new_data['imageHeight'] = transformed_image.shape[0]
                new_data['imageWidth'] = transformed_image.shape[1]

                # 更新 shapes 坐标
                new_shapes = []
                for box, label in zip(transformed_bboxes, transformed_category_ids):
                    # Albumentations 返回的是 [x_min, y_min, x_max, y_max]
                    new_shape = {
                        "label": label,
                        "points": [[box[0], box[1]], [box[2], box[3]]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    new_shapes.append(new_shape)

                new_data['shapes'] = new_shapes

                # 写入新的 JSON 文件
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"处理 {filename} 的第 {i} 次增强时出错: {e}")

    print(f"处理完成！所有增强后的文件已保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()