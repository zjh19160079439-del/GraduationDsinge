import cv2

video_path = r'D:\Adatasets\xiaocanVideo\goods.mp4'  # 视频地址
output_path = r'D:\Adatasets\GoodsDatasets/'  # 输出文件夹
interval = 10  # 每间隔10帧取一张图片，


if __name__ == '__main__':
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            if num % interval == 1:
                file_name = '%03d' % num
                cv2.imwrite(str(file_name) + '.jpg', frame)
                # 00000111.jpg 代表第111帧
                cv2.waitKey(1)
            num += 1

        else:
            break
