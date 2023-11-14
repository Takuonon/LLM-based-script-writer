import cv2
import numpy as np
from ultralytics import YOLO


def detect_figure(image_path):
    model = YOLO("ultralytics/runs/detect/train2/weights/best.pt")
    results = model(image_path, save_crop=True, save=True, conf=0.7)
    boxes = []
    for r in results:
        # gpuで推論するときはtesorをGPUからcpuに移動させる必要があるかも
        boxes.append(r.boxes.xywhn.numpy())

    return boxes


# この関数は処理が結構脳筋になってます、ごめんなさい、、、
def fill_boxes_white(image_path, boxes):
    # numpy配列をPythonのリストに変換
    boxes_list = [box.tolist() for box in boxes][0]

    # 画像を読み込む
    image = cv2.imread(image_path)
    # 各ボックスに対して、画像領域を白で塗りつぶす
    for box in boxes_list:
        # YOLOの座標系とOpenCVの座標系の間で置換が必要
        # YOLOの座標系を受け取る
        x_center, y_center, w, h = box

        # 画像の実際の幅と高さ
        image_width, image_height = image.shape[1], image.shape[0]

        # x_center, y_center, w, hをピクセルの座標に変換
        x_center = int(x_center * image_width)
        y_center = int(y_center * image_height)
        w = int(w * image_width)
        h = int(h * image_height)

        # OpenCVの座標系に変換
        x_top_left = x_center - w // 2
        y_top_left = y_center - h // 2

        image[y_top_left : y_top_left + h, x_top_left : x_top_left + w] = [255, 255, 255]  # 白で塗りつぶす

    # 画像を保存
    cv2.imwrite(image_path, image)


# image_path = "example/Computer Architectures-202.jpg"
# a = detect_figure(image_path)
# print("boxes:", a)
# runs/detect/predictの中に保存される。
# もし2周以上回った場合にはruns/detect/predict2のように保存される
# 取得後のboxをxywh形式で取得
