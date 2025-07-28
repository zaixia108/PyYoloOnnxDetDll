import time

import cv2

import YoloOnnxDet

std = YoloOnnxDet.ST_Detector(
    model_path='src/atkfp16.onnx',
    names=[1, 2, 3, 4, 5, 6, 7],
    conf_threshold=0.3,
    iou_threshold=0.5,
)

RANGE = 200

std.warm_up()
img = cv2.imread('src/img_5.png')

t1 = time.time()
result = std.detect(img)
print(result)
t2 = time.time()
print(f"Single-thread detection time: {(t2 - t1) * 1000:.2f} ms 1 image")

mtd = YoloOnnxDet.MT_Detector(
    model_path='src/atkfp16.onnx',
    names=[1, 2, 3, 4, 5, 6, 7],
    conf_threshold=0.3,
    iou_threshold=0.5,
    workers=4
)
mtd.warm_up()
pending_list = []
t1 = time.time()
for i in range(RANGE):
    t3 = time.time()
    result_mt = mtd.submit_task(img)
    t4 = time.time()
    pending_list.append(result_mt)
while True:
    finish = [mtd.get_task_status(p) for p in pending_list]
    if all(finish):
        for pl in pending_list:
            result = mtd.get_result(pl)
        break
t2 = time.time()
print(f"Multi-thread detection time: {(t2 - t1) * 1000:.2f} ms {RANGE} image")
mtd.cleanup()