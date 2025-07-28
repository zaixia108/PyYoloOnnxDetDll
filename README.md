# YoloOnnxDet

## 项目简介
YoloOnnxDet 是一个基于 ONNX 和 C++ DLL 的目标检测 Python 封装，支持高效的图片检测，支持单线程和多线程推理。

## 依赖环境
- Python 3.8+
- numpy
- opencv-python
- ONNX 格式的模型文件

安装依赖：
```bash
pip install numpy opencv-python
```

## 快速开始

1. 准备好 `OnnxDet.dll` 和 ONNX 模型文件。
2. 单线程检测示例：

```python
from YoloOnnxDet import ST_Detector
import cv2

det = ST_Detector("your_model.onnx", names=["class1", "class2"], conf_threshold=0.3, iou_threshold=0.5)
image = cv2.imread("your_image.png")
boxes, scores, class_ids = det.detect(image)
print(boxes, scores, class_ids)
```

3. 多线程检测示例（需先启用多线程功能）：

```python
from YoloOnnxDet import MT_Detector, DevFeature
import cv2

DevFeature.EnableMultithreaded = True
names = ["class1", "class2"]
det = MT_Detector("your_model.onnx", names=names, conf_threshold=0.3, iou_threshold=0.5, workers=4)
image = cv2.imread("your_image.png")
task_id = det.submit_task(image)
det.wait_for_task_completion(task_id)
results = det.get_result(task_id)
print(results)
```

## 资源监控
可通过 `get_pool_status()` 获取多线程池状态，包括队列长度、活跃线程数等。

## 资源释放
建议在检测结束后调用 `cleanup()` 方法释放资源。

## 其他说明
- 多线程检测为开发功能，稳定性有限。
- 支持自定义类别名（names 参数）。
- 需保证 DLL 文件与 Python 文件在同一目录或已配置环境变量。
