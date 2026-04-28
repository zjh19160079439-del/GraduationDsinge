import os
import torch
import torch.nn as nn
import torchvision
import xml.etree.ElementTree as ET
import json
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.transforms import functional as F


# ==========================================
# 第2步：准备数据集与标注格式 (VOC XML / TXT / JSON)
# ==========================================
class SilkwormDataset(Dataset):
    def __init__(self, root_dir, transforms=None, annot_format='xml'):
        """
        假设目录结构:
        root_dir/
            images/
                train/
            labels/
                train/
        :param annot_format: 标注文件格式，支持 'xml', 'txt' (YOLO), 'json' (LabelMe)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.annot_format = annot_format.lower()
        self.img_dir = os.path.join(root_dir, "images", "train")
        self.label_dir = os.path.join(root_dir, "labels", "train")
        self.imgs = list(sorted(os.listdir(self.img_dir)))

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = []
        labels = []

        # 1. 解析 XML 格式 (VOC)
        if self.annot_format == 'xml':
            label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + ".xml")
            if os.path.exists(label_path):
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # 类别1代表小蚕

        # 2. 解析 TXT 格式 (YOLO)
        elif self.annot_format == 'txt':
            label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO格式: class x_center y_center width height (归一化0~1)
                            x_center, y_center, w, h = map(float, parts[1:5])
                            # 转换为绝对坐标 xmin, ymin, xmax, ymax
                            xmin = (x_center - w / 2) * img_w
                            ymin = (y_center - h / 2) * img_h
                            xmax = (x_center + w / 2) * img_w
                            ymax = (y_center + h / 2) * img_h
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(1)

        # 3. 解析 JSON 格式 (LabelMe)
        elif self.annot_format == 'json':
            label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + ".json")
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for shape in data.get('shapes', []):
                        points = shape['points']
                        # 提取左上角和右下角坐标
                        xmin = min(points[0][0], points[1][0])
                        ymin = min(points[0][1], points[1][1])
                        xmax = max(points[0][0], points[1][0])
                        ymax = max(points[0][1], points[1][1])
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(1)

        # 防止有些图像没有目标导致张量维度报错
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 第1步 & 第3步：网络结构、配置模型与预设默认框
# ==========================================
class MultiScaleMobileNetV2(nn.Module):
    """
    包装 MobileNetV2，添加额外的下采样层 (Extra Blocks)
    使其输出 6 个不同尺度的特征图，以适配 SSD 多尺度检测。
    """

    def __init__(self):
        super().__init__()
        # 1. 基础骨干网络 (MobileNetV2)
        mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
        self.features = mobilenet

        # 2. SSD Extra Blocks (额外 5 个尺度的下采样特征图)
        self.extra = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1280, 512, 1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU())
        ])

        # PyTorch SSD 规定 backbone 必须包含 out_channels 属性，表示每个特征图的通道数
        self.out_channels = [1280, 512, 256, 256, 128, 128]

    def forward(self, x):
        out = OrderedDict()

        # 第一层特征图 (由原生 MobileNetV2 输出)
        x = self.features(x)
        out["0"] = x

        # 依次通过 Extra Blocks 提取更深层特征图
        for i, block in enumerate(self.extra):
            x = block(x)
            out[str(i + 1)] = x

        return out


def create_ssd_mobilenet_v2(num_classes=2):  # 背景(0) + 小蚕(1)
    """
    构建基于 MobileNetV2 的 SSD 检测模型。
    """
    # 实例化自定义的多尺度 MobileNetV2
    backbone = MultiScaleMobileNetV2()

    # 构建在 6 个不同尺度特征图上的锚框 (Anchor / Default Boxes)
    # 【修复重点】：aspect_ratios 长度为 6，scales 的长度必须是 7 (需补充最后的上限 scale 1.05)
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],  # 已添加 1.05 解决 IndexError
        steps=[16, 32, 64, 128, 256, 512]
    )

    # 组装 SSD 模型
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(300, 300)  # 输入图像分辨率
    )
    return model


# 图像预处理
def transform(img):
    return F.to_tensor(img)


def collate_fn(batch):
    return tuple(zip(*batch))


# ==========================================
# 第4步 & 第5步：匹配策略、训练损失与模型训练
# ==========================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据集
    # 👇【使用说明】：在这里修改 annot_format 参数为 'txt' 或 'json' 即可直接读取对应格式！
    dataset = SilkwormDataset(root_dir="D:\GraduationDsign\datasets\yolo_dataset", transforms=transform, annot_format='txt')
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # 2. 初始化模型
    model = create_ssd_mobilenet_v2(num_classes=2)
    model.to(device)

    # 3. 优化器与学习率调度 (SGD + CosineAnnealingLR)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # 余弦退火衰减，T_max=100 (代表100个epoch周期)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    num_epochs = 100
    model.train()  # 切换为训练模式

    print("开始训练 SSD-MobileNetV2 基线模型...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        # 添加 enumerate 获取 step 索引
        for step, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            loss_dict = model(images, targets)

            # 总损失 L = L_conf + L_loc
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播与优化
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # 【新增】：每隔几个 Batch 打印一次当前进度，证明程序还在动！
            if step % 2 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{len(data_loader)}], Batch Loss: {losses.item():.4f}")

        # 更新学习率
        scheduler.step()

        # 【修改】：将 % 10 去掉，保证每个 Epoch 结束都打印一次平均 Loss
        print(
            f"====> Epoch [{epoch + 1}/{num_epochs}] 结束, 平均 Loss: {epoch_loss / len(data_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存训练好的模型权重
    torch.save(model.state_dict(), "ssd_mobilenet_v2_silkworm.pth")
    print("训练完成并保存模型！")


# ==========================================
# 第6步：推理与后处理（NMS）
# ==========================================
def infer_and_count(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_ssd_mobilenet_v2(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 切换到推理模式

    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 获取预测结果
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    confidence_threshold = 0.5
    iou_threshold = 0.45

    # 1. 置信度过滤
    keep_conf = scores > confidence_threshold
    boxes = boxes[keep_conf]
    scores = scores[keep_conf]

    # 2. 非极大值抑制 (NMS)
    keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
    final_boxes = boxes[keep_nms]
    final_scores = scores[keep_nms]

    # 第7步：计数与结果输出
    silkworm_count = len(final_boxes)
    print(f"--- 推理结果 ---")
    print(f"图像: {image_path}")
    print(f"检测到的小蚕数量: {silkworm_count}")

    return final_boxes, final_scores


if __name__ == "__main__":
    # 取消注释以运行训练
    train_model()

    # 模拟推理（请确保路径下有对应图像与模型权重）
    # infer_and_count("dataset/images/test/sample_001.jpg", "ssd_mobilenet_v2_silkworm.pth")