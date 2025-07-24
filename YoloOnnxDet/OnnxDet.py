import ctypes
import os.path
from importlib.resources import files as resource_path
import cv2
import numpy as np


class OnnxDetector:
    def __init__(self, model_path, names, conf_threshold=0.3, iou_threshold=0.5):
        """
        初始化ONNX检测器
        :param model_path: ONNX模型文件路径
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IoU阈值
        """
        dll_path = str(resource_path('YoloOnnxDet').joinpath('OnnxDet.dll'))
        self.lib = ctypes.WinDLL(dll_path)

        # 设置函数原型
        self.lib.CreateDetector.restype = ctypes.c_void_p

        self.lib.DestroyDetector.argtypes = [ctypes.c_void_p]

        self.lib.InitDetector.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_float
        ]
        self.lib.InitDetector.restype = ctypes.c_bool

        self.lib.Detect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.Detect.restype = ctypes.c_bool

        self.lib.ReleaseResults.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int)
        ]

        # 创建检测器实例
        self.detector = self.lib.CreateDetector()
        if not self.detector:
            raise RuntimeError("创建检测器失败")

        if os.path.exists(model_path):
            pass
        else:
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        # 初始化检测器
        model_path_bytes = model_path.encode('utf-8')
        result = self.lib.InitDetector(
            self.detector,
            model_path_bytes,
            ctypes.c_float(conf_threshold),
            ctypes.c_float(iou_threshold)
        )

        if not result:
            self.lib.DestroyDetector(self.detector)
            raise RuntimeError("初始化检测器失败")

        if type(names) == list:
            self.names = [str(name) for name in names]
        elif type(names) == str:
            with open(names, 'r', encoding='utf-8') as f:
                self.names = [f.strip() for f in f.readlines()]

    def __del__(self):
        """析构函数，释放检测器资源"""
        if hasattr(self, 'lib') and hasattr(self, 'detector') and self.detector:
            self.lib.DestroyDetector(self.detector)
            self.detector = None

    def detect(self, image):
        """
        执行目标检测
        :param image: OpenCV格式的图像 (BGR)
        :return: (boxes, scores, class_ids) 元组
        """
        # 确保图像是BGR格式的
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 获取图像信息
        height, width, channels = image.shape

        # 准备输出参数
        p_boxes = ctypes.POINTER(ctypes.c_float)()
        p_scores = ctypes.POINTER(ctypes.c_float)()
        p_classes = ctypes.POINTER(ctypes.c_int)()
        count = ctypes.c_int(0)

        # 获取图像数据指针
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        # 执行检测
        result = self.lib.Detect(
            self.detector,
            img_data,
            width,
            height,
            channels,
            ctypes.byref(p_boxes),
            ctypes.byref(p_scores),
            ctypes.byref(p_classes),
            ctypes.byref(count)
        )

        if not result:
            raise RuntimeError("检测过程发生错误")

        # 处理检测结果
        num_detections = count.value
        boxes = []
        scores = []
        class_ids = []

        if num_detections > 0:
            # 将C指针转换为NumPy数组
            boxes_array = np.zeros((num_detections, 4), dtype=np.float32)
            for i in range(num_detections):
                for j in range(4):
                    boxes_array[i, j] = p_boxes[i * 4 + j]

            scores = np.array([p_scores[i] for i in range(num_detections)], dtype=np.float32)
            class_ids = np.array([p_classes[i] for i in range(num_detections)], dtype=np.int32)

            # 释放C++端分配的内存
            self.lib.ReleaseResults(p_boxes, p_scores, p_classes)

            boxes = boxes_array
        result = {}
        for i in self.names:
            result[i] = []
        for i in range(len(boxes)):
            p1 = int(boxes[i][0])
            p2 = int(boxes[i][1])
            p3 = int(boxes[i][2])
            p4 = int(boxes[i][3])
            lt = (p1, p2)
            rb = (p3, p4)
            data = {
                'confidence': round(scores[i], 2),
                'box': [lt, rb],
                'center': (int((lt[0] + rb[0]) // 2), int((lt[1] + rb[1]) // 2)),
            }
            result[self.names[class_ids[i]]].append(data)
        return result