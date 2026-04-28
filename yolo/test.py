from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') # 加载姿态模型
# 关键：必须设置 save=True 才会保存图片到磁盘
results = model.predict('ultralytics/KeyPose/code/capture_20260326_153653_000.jpg', save=True)

# 如果你想直接在代码里查看保存路径
for r in results:
    print(f"预测图片保存在: {r.save_dir}")