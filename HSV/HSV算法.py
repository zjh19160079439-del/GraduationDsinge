import cv2
import numpy as np


def create_dummy_image():
    """
    辅助函数：生成一张带噪点、包含独立和粘连小蚕的虚拟测试图像，
    以便在没有真实图片时也能直接运行测试。
    """
    # 创建暗绿色背景 (模拟桑叶)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = [50, 100, 50]

    # 浅黄白色 (模拟小蚕颜色，BGR格式)
    color = (180, 230, 240)

    # 1. 画一个独立的小蚕
    cv2.circle(img, (80, 80), 12, color, -1)
    # 2. 画两个粘连的小蚕
    cv2.circle(img, (200, 200), 15, color, -1)
    cv2.circle(img, (220, 215), 13, color, -1)
    # 3. 画三个严重粘连的小蚕 (测试距离变换和分水岭)
    cv2.circle(img, (320, 100), 14, color, -1)
    cv2.circle(img, (340, 110), 12, color, -1)
    cv2.circle(img, (325, 125), 13, color, -1)
    # 4. 画一个极小的噪点 (测试面积过滤)
    cv2.circle(img, (100, 300), 2, color, -1)

    # 添加高斯噪声
    noise = np.random.randint(0, 30, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def count_silkworms(img_path=None):
    # ==========================================
    # 第1步：图像预处理
    # ==========================================
    if img_path:
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 找不到图片 {img_path}")
            return
    else:
        print("未提供图片路径，使用自动生成的虚拟测试图...")
        img = create_dummy_image()

    output_img = img.copy()  # 用于最后绘制结果

    # 高斯平滑去噪并转换到HSV色彩空间
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 第2步：HSV颜色空间阈值分割
    # ==========================================
    # 提取浅黄白色小蚕区域
    lower = np.array([20, 30, 150])
    upper = np.array([35, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # ==========================================
    # 第3步：形态学操作去噪
    # ==========================================
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 开运算：去除外部孤立小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 闭运算：填补内部小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ==========================================
    # 第4步：距离变换（关键步骤）
    # ==========================================
    # 计算每个前景像素到背景的最短距离
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 为了直观显示距离变换效果（可选），可以做个归一化副本，但不影响后续计算
    dist_display = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # ==========================================
    # 第5步：生成前景/背景标记（改进点）
    # ==========================================
    # 阈值化获取确定的前景区域 (找出粘连个体的各自中心点)
    _, fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    fg = np.uint8(fg)

    # 膨胀掩码获取确定的背景区域
    bg = cv2.dilate(mask, kernel, iterations=3)

    # 计算未知区域 (即分水岭算法去寻找边界的区域)
    unknown = cv2.subtract(bg, fg)

    # 连通域标记：为每个独立的前景中心分配一个唯一的ID (从1开始递增)
    _, markers = cv2.connectedComponents(fg)

    # 分水岭算法中，背景需要被标记为1，未知区域标记为0
    markers = markers + 1
    markers[unknown == 255] = 0

    # ==========================================
    # 第6步：执行分水岭分割
    # ==========================================
    # 使用修改后的标记执行分水岭算法，边界处的标记会被置为 -1
    markers = cv2.watershed(img, markers)

    # 将找到的分水岭边界在原图上标红
    output_img[markers == -1] = [0, 0, 255]

    # ==========================================
    # 第7步：轮廓提取与计数
    # ==========================================
    silkworm_count = 0

    # 遍历所有生成的标记 (忽略背景标记1和边界标记-1)
    unique_markers = np.unique(markers)
    for marker_value in unique_markers:
        if marker_value == 1 or marker_value == -1:
            continue

        # 为当前标记提取独立掩码
        obj_mask = np.zeros_like(mask)
        obj_mask[markers == marker_value] = 255

        # 提取当前独立区域的轮廓
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 根据先验知识的面积阈值进行过滤
            if 15 <= area <= 5000:  # 为了适配虚拟大图，这里上限放宽到5000，实际使用时可改回500
                silkworm_count += 1

                # 绘制有效轮廓（绿色包裹线）
                cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)

                # 在轮廓中心打个蓝点标记
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(output_img, (cx, cy), 3, (255, 0, 0), -1)

    print(f"--> 分割并过滤完成，最终统计到的小蚕数量为: {silkworm_count} 只\n")

    # ==========================================
    # 结果展示
    # ==========================================
    cv2.imshow("1. Original Image", img)
    cv2.imshow("2. HSV Mask (After Morph)", mask)
    cv2.imshow("3. Distance Transform", dist_display)
    cv2.imshow("4. Watershed Result & Count", output_img)

    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 如果有真实图片，请传入图片路径，例如：
    count_silkworms(r"D:\GraduationDsign\datasets\yolo_dataset\images\train\00000001_aug_0.jpg")

    # 默认调用（将自动生成内部测试图来演示效果）
    # count_silkworms()