import cv2
from ultralytics import YOLO
import time

# ================= 配置区域 =================
# 替换为你训练好的权重路径，通常在 runs/detect/train/weights50/best.pt
MODEL_PATH = "runs/detect/train4/weights50/best.pt"
# 替换为你要检测的视频路径
VIDEO_PATH = r"D:\datasets\xiaocanVideo\1 (1).mp4"
# 检测后保存的视频路径
OUTPUT_PATH = r"D:\datasets\xiaocanVideo\output_video2.mp4"
# 置信度阈值，低于此分数的预测框将被过滤
CONF_THRESH = 0.5


def main():
    # 1. 加载训练好的 YOLOv8 模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败，请检查路径是否正确。错误信息: {e}")
        return

    # 2. 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频文件: {VIDEO_PATH}，请检查路径。")
        return

    # 获取原视频的宽度、高度和帧率，用于保存预测后的视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"视频分辨率: {width}x{height}, 帧率: {fps}")

    # 3. 设置视频写入器 (VideoWriter)
    # 使用 mp4v 编码器保存为 mp4 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("开始进行视频检测，按 'q' 键可提前退出显示...")

    # 4. 逐帧读取并检测
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("视频读取完毕或由于报错中断。")
            break

        # 使用模型对当前帧进行推理预测
        # conf参数用于过滤低置信度的框，verbose=False用来关闭控制台逐帧打印信息
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)

        # 取出第一张图（因为我们只传入了一帧）的预测结果
        result = results[0]

        # result.plot() 会自动将预测框、类别标签和置信度画在图像上，并返回一个 numpy 数组
        annotated_frame = result.plot()

        # 选做：如果你想在画面左上角加上蚕的数量统计，可以取消下面两行的注释
        silkworm_count = len(result.boxes)
        cv2.putText(annotated_frame, f"Count: {silkworm_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 写入保存的视频文件中
        out.write(annotated_frame)

        # 实时显示检测画面 (如果是在没有图形界面的服务器上运行，请注释掉下面三行)
        cv2.imshow("Silkworm Detection", annotated_frame)

        # 按 'q' 键可以随时退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户手动中断检测。")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧...")

    # 5. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"检测完成！共处理 {frame_count} 帧，耗时 {end_time - start_time:.2f} 秒。")
    print(f"结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()