import sys
import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QCheckBox,
                             QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO


# ==========================================
# 1. 图像处理类 (ImageProcessor)
# ==========================================
class ImageProcessor:
    @staticmethod
    def draw_boxes(image, boxes, show_label=True):
        """在图像副本上绘制检测框"""
        img_drawn = image.copy()
        count = len(boxes)
        for box in boxes:
            # YOLOv8 boxes.xyxy 已经自动还原到了原图坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 绘制红色矩形框 (BGR: 0, 0, 255)，线宽 2
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # 如果勾选了显示标签，绘制半透明背景条和置信度
            if show_label:
                label = f"{conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_drawn, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(img_drawn, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img_drawn, count


# ==========================================
# 2. 检测器类 (YOLOv8Detector)
# ==========================================
class YOLOv8Detector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, image, conf_thresh):
        """执行推理，返回检测结果"""
        # YOLOv8自带预处理，且我们传入实际图像和阈值
        results = self.model.predict(source=image, conf=conf_thresh, verbose=False)
        return results[0].boxes


# ==========================================
# 3. 工作线程类 (WorkerThread) - 防止UI卡死
# ==========================================
class WorkerThread(QThread):
    # 自定义信号：发射处理后的原图(BGR)、结果图(BGR)和计数值
    finished_signal = pyqtSignal(np.ndarray, np.ndarray, int)

    def __init__(self, detector, processor, image, conf, show_label):
        super().__init__()
        self.detector = detector
        self.processor = processor
        self.image = image
        self.conf = conf
        self.show_label = show_label

    def run(self):
        # 1. 模型推理
        boxes = self.detector.detect(self.image, self.conf)
        # 2. 图像后处理（绘制框）
        result_img, count = self.processor.draw_boxes(self.image, boxes, self.show_label)
        # 3. 发射结果更新UI
        self.finished_signal.emit(self.image, result_img, count)


# ==========================================
# 4. 主窗口 / 主控制器 (MainController)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小蚕智能识别系统")
        self.resize(1200, 700)

        # 初始化核心组件 (请将 'best.pt' 替换为你实际的模型路径)
        try:
            self.detector = YOLOv8Detector(r'D:\Program Files\student code\ultralytics-8.2.26\runs\detect\SilkTrain01\weights\best.pt')
        except Exception as e:
            QMessageBox.warning(self, "警告", "未找到 best.pt 模型，请确保路径正确！")
            self.detector = None

        self.processor = ImageProcessor()

        # 状态变量
        self.current_image = None
        self.result_image = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.cap = None
        self.history_data = []

        self.init_ui()

    def init_ui(self):
        # 核心部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==================== 左侧：控制与输入区 ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(250)

        # 输入源控制
        self.btn_open_img = QPushButton("打开图片")
        self.btn_open_cam = QPushButton("开启相机")

        # 置信度滑块 (取值范围 10-90，代表 0.1-0.9)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(50)
        self.lbl_conf = QLabel("置信度阈值: 0.5")
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"置信度阈值: {v / 100:.2f}"))

        # 标签复选框
        self.cb_show_label = QCheckBox("显示标签")
        self.cb_show_label.setChecked(True)

        # 功能按钮
        self.btn_start = QPushButton("开始识别")
        self.btn_save = QPushButton("保存图片")
        self.btn_export = QPushButton("导出数据")

        # 结果显示
        self.lbl_result_count = QLabel("识别结果：0头")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.lbl_result_count.setFont(font)
        self.lbl_result_count.setStyleSheet("color: red; margin-top: 20px;")

        left_layout.addWidget(self.btn_open_img)
        left_layout.addWidget(self.btn_open_cam)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.lbl_conf)
        left_layout.addWidget(self.slider_conf)
        left_layout.addWidget(self.cb_show_label)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(self.btn_export)
        left_layout.addWidget(self.lbl_result_count)
        left_layout.addStretch()

        # ==================== 中部：图像显示区 ====================
        mid_panel = QWidget()
        mid_layout = QHBoxLayout(mid_panel)

        self.lbl_img_original = QLabel("原始图像")
        self.lbl_img_original.setAlignment(Qt.AlignCenter)
        self.lbl_img_original.setStyleSheet("background-color: #eee; border: 1px solid #ccc;")

        self.lbl_img_result = QLabel("结果图像")
        self.lbl_img_result.setAlignment(Qt.AlignCenter)
        self.lbl_img_result.setStyleSheet("background-color: #eee; border: 1px solid #ccc;")

        mid_layout.addWidget(self.lbl_img_original)
        mid_layout.addWidget(self.lbl_img_result)

        # ==================== 右侧：历史记录区 ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(250)

        self.table_history = QTableWidget(0, 3)
        self.table_history.setHorizontalHeaderLabels(["文件名/来源", "时间", "计数"])
        self.table_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(QLabel("历史记录"))
        right_layout.addWidget(self.table_history)

        # 将三部分加入主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(mid_panel)
        main_layout.addWidget(right_panel)

        # 绑定信号与槽
        self.btn_open_img.clicked.connect(self.open_image)
        self.btn_open_cam.clicked.connect(self.toggle_camera)
        self.btn_start.clicked.connect(self.process_image)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_export.clicked.connect(self.export_data)

    # ---------------- 业务逻辑 ----------------
    def cvimg_to_qt(self, cv_img):
        """将 OpenCV 的 BGR 图像转换为 PyQt 的 QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img)

    def display_images(self, orig_img, res_img=None):
        """在界面上按比例缩放显示图像"""
        # 获取 Label 的尺寸
        lbl_w, lbl_h = self.lbl_img_original.width(), self.lbl_img_original.height()

        pix_orig = self.cvimg_to_qt(orig_img)
        self.lbl_img_original.setPixmap(pix_orig.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if res_img is not None:
            pix_res = self.cvimg_to_qt(res_img)
            self.lbl_img_result.setPixmap(pix_res.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def open_image(self):
        # 关闭可能正在运行的相机
        if self.camera_timer.isActive():
            self.toggle_camera()

        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.source_name = file_path.split('/')[-1]
            self.display_images(self.current_image)
            self.lbl_img_result.setText("等待识别...")

    def toggle_camera(self):
        if not self.camera_timer.isActive():
            self.cap = cv2.VideoCapture(0)  # 0为默认摄像头
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头！")
                return
            self.source_name = "Camera Stream"
            self.camera_timer.start(30)  # 约 30 FPS
            self.btn_open_cam.setText("关闭相机")
        else:
            self.camera_timer.stop()
            self.cap.release()
            self.btn_open_cam.setText("开启相机")
            self.lbl_img_original.setText("原始图像")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_image = frame
            self.display_images(self.current_image)

    def process_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先载入图片或开启相机！")
            return

        if self.detector is None:
            QMessageBox.warning(self, "提示", "模型未加载成功，无法识别。")
            return

        self.btn_start.setEnabled(False)
        self.btn_start.setText("识别中...")

        # 获取参数设置
        conf = self.slider_conf.value() / 100.0
        show_label = self.cb_show_label.isChecked()

        # 启动工作线程
        self.thread = WorkerThread(self.detector, self.processor, self.current_image, conf, show_label)
        self.thread.finished_signal.connect(self.on_process_finished)
        self.thread.start()

    def on_process_finished(self, orig_img, res_img, count):
        self.result_image = res_img
        self.display_images(orig_img, res_img)
        self.lbl_result_count.setText(f"识别结果：{count}头")

        # 恢复按钮状态
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始识别")

        # 添加历史记录
        self.add_history(self.source_name, count)

    def add_history(self, source, count):
        now = datetime.now().strftime("%H:%M:%S")
        self.history_data.append({"Source": source, "Time": now, "Count": count})

        row = self.table_history.rowCount()
        self.table_history.insertRow(row)
        self.table_history.setItem(row, 0, QTableWidgetItem(source))
        self.table_history.setItem(row, 1, QTableWidgetItem(now))
        self.table_history.setItem(row, 2, QTableWidgetItem(str(count)))
        self.table_history.scrollToBottom()

    def save_result(self):
        if self.result_image is None:
            QMessageBox.warning(self, "提示", "当前没有可保存的识别结果！")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "result.jpg", "JPEG (*.jpg);;PNG (*.png)")
        if save_path:
            cv2.imwrite(save_path, self.result_image)
            QMessageBox.information(self, "成功", "图片保存成功！")

    def export_data(self):
        if not self.history_data:
            QMessageBox.warning(self, "提示", "当前没有历史数据可导出！")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "导出数据", "silkworm_history.xlsx", "Excel (*.xlsx)")
        if save_path:
            df = pd.DataFrame(self.history_data)
            df.to_excel(save_path, index=False)
            QMessageBox.information(self, "成功", "数据导出成功！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())