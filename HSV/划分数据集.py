import os
import json
import random
import shutil

# ================= 配置区域 =================
# 你刚才数据增强后保存的文件夹路径
INPUT_DIR = r"/yolo_datasets"
# 转换后 YOLO 格式数据集的保存路径
OUTPUT_DIR = "yolo_dataset/"
# 训练集所占比例 (例如 0.8 表示 80% 训练集，20% 验证集)
TRAIN_RATIO = 0.8


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_yolo_format(bbox, img_w, img_h):
    """
    将 LabelMe 的 bbox [x_min, y_min, x_max, y_max]
    转换为 YOLO 的 [center_x, center_y, width, height] 并归一化
    """
    x_min, y_min, x_max, y_max = bbox

    # 计算中心点坐标和宽高
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # 归一化 (除以图片的宽和高)
    center_x /= img_w
    center_y /= img_h
    width /= img_w
    height /= img_h

    return center_x, center_y, width, height


def main():
    # 1. 创建 YOLO 目录结构
    images_train_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    images_val_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    labels_train_dir = os.path.join(OUTPUT_DIR, 'labels', 'train')
    labels_val_dir = os.path.join(OUTPUT_DIR, 'labels', 'val')

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        ensure_dir(d)

    # 2. 获取所有图片和对应的 json
    all_files = os.listdir(INPUT_DIR)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    valid_pairs = []
    classes_set = set()

    print("正在扫描数据集并提取类别信息...")
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        json_file = base_name + '.json'
        json_path = os.path.join(INPUT_DIR, json_file)

        if os.path.exists(json_path):
            valid_pairs.append((img_file, json_file))
            # 读取 json 收集所有的类别名
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    classes_set.add(shape['label'])

    # 创建类别名到索引的映射 (例如: {'silkworm': 0})
    classes_list = sorted(list(classes_set))
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes_list)}

    print(f"发现的类别: {class_to_id}")
    print(f"有效数据对数量: {len(valid_pairs)}")

    # 3. 打乱数据并划分训练集/验证集
    random.seed(42)  # 固定随机种子以保证每次划分结果一致
    random.shuffle(valid_pairs)

    train_size = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:]

    print(f"划分结果: 训练集 {len(train_pairs)} 张, 验证集 {len(val_pairs)} 张")

    # 4. 处理并复制文件
    def process_dataset(pairs, target_images_dir, target_labels_dir):
        for img_file, json_file in pairs:
            # 复制图片
            src_img_path = os.path.join(INPUT_DIR, img_file)
            dst_img_path = os.path.join(target_images_dir, img_file)
            shutil.copy(src_img_path, dst_img_path)

            # 读取并转换 JSON 到 TXT
            json_path = os.path.join(INPUT_DIR, json_file)
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(target_labels_dir, txt_file)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_w = data['imageWidth']
            img_h = data['imageHeight']

            with open(txt_path, 'w', encoding='utf-8') as f_txt:
                for shape in data['shapes']:
                    if shape['shape_type'] == 'rectangle':
                        cls_id = class_to_id[shape['label']]
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]

                        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                        yolo_bbox = convert_to_yolo_format(bbox, img_w, img_h)

                        # 写入一行: class_id center_x center_y w h
                        line = f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n"
                        f_txt.write(line)

    print("正在处理训练集...")
    process_dataset(train_pairs, images_train_dir, labels_train_dir)

    print("正在处理验证集...")
    process_dataset(val_pairs, images_val_dir, labels_val_dir)

    # 5. 生成 data.yaml 文件
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # 注意：这里的路径建议在实际训练时根据服务器或本地环境修改为绝对路径
        f.write(f"train: ./images/train\n")
        f.write(f"val: ./images/val\n")
        f.write(f"\n")
        f.write(f"nc: {len(classes_list)}\n")
        f.write(f"names: {classes_list}\n")

    print(f"数据转换并划分完成！请查看目录: {OUTPUT_DIR}")
    print(f"生成的 data.yaml 已保存在: {yaml_path}")


if __name__ == "__main__":
    main()