import cv2
import numpy as np


# ==========================================
# 核心算法：刚才写好的分水岭分割函数
# ==========================================
def count_adhering_silkworms(original_img, binary_mask, area_min=15, area_max=500):
    result_img = original_img.copy()

    # 1. 获取确定的背景区域
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=2)

    # 2. 计算距离变换
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 3. 低通滤波与峰值抑制
    dist_smooth = cv2.GaussianBlur(dist_transform, (5, 5), 0)

    # 这里的窗口大小(15)决定了最小蚕的间距，如果图片分辨率很高，可以适当调大
    min_dist_window = 15
    kernel_max = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_dist_window, min_dist_window))
    local_max = cv2.dilate(dist_smooth, kernel_max)

    peak_mask = (dist_smooth == local_max) & (dist_smooth > 0.1 * dist_smooth.max())
    sure_fg = np.uint8(peak_mask) * 255
    sure_fg = cv2.dilate(sure_fg, np.ones((3, 3), np.uint8), iterations=1)

    # 4. 确定未知区域并生成标记
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. 执行分水岭算法
    markers = cv2.watershed(original_img, markers)

    # 6. 提取轮廓并计数
    segmented_objects = np.zeros_like(binary_mask, dtype=np.uint8)
    segmented_objects[markers > 1] = 255

    contours, _ = cv2.findContours(segmented_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min <= area <= area_max:
            valid_contours.append(cnt)
            # 在结果图上用绿色画出有效轮廓
            cv2.drawContours(result_img, [cnt], 0, (0, 255, 0), 2)

    count = len(valid_contours)
    return count, result_img


# ==========================================
# 主程序：使用流程
# ==========================================
if __name__ == "__main__":
    # 1. 读取原始图像
    img_path = r"D:\GraduationDsign\datasets\yolo_dataset\images\train\00000136_aug_0.jpg"  # 替换为你的图片路径
    img = cv2.imread(img_path)

    if img is None:
        print(f"错误: 无法读取图片 {img_path}，请检查路径。")
        exit()

    # 2. 将图像转换到 HSV 色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 设定蚕的 HSV 颜色范围 (这一步非常关键，需要根据你的实际图片进行调整！)
    # 注意：这里的数值只是示例，假设蚕偏黄绿色。你需要用拾色器获取真实范围。
    lower_color = np.array([43, 0, 29])
    upper_color = np.array([142, 255, 255])

    # 4. 根据颜色范围提取掩膜 (二值化)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 5. 初步去噪 (形态学开运算：先腐蚀后膨胀，去除小的噪点)
    kernel_denoise = np.ones((3, 3), np.uint8)
    mask_denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_denoise, iterations=1)

    # 6. 调用改进的分水岭算法进行粘连分割与计数
    # 注意：area_min 和 area_max 需要根据你图片中小蚕的实际像素面积来调整
    silkworm_count, result_image = count_adhering_silkworms(
        original_img=img,
        binary_mask=mask_denoised,
        area_min=20,  # 忽略面积小于20的噪点
        area_max=2000  # 忽略面积大于2000的杂质 (数值视原图分辨率而定)
    )

    # 7. 打印结果
    print(f"✅ 算法计算得出的小蚕数量为: {silkworm_count} 只")

    # 8. 可视化展示结果
    # 缩小图片以便在屏幕上完整显示 (可选)
    display_scale = 0.5
    h, w = img.shape[:2]
    result_display = cv2.resize(result_image, (int(w * display_scale), int(h * display_scale)))
    mask_display = cv2.resize(mask_denoised, (int(w * display_scale), int(h * display_scale)))

    cv2.imshow("1. Denoised Mask", mask_display)
    cv2.imshow("2. Watershed Result", result_display)

    print("按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()