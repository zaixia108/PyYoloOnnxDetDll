import logging
import time
import traceback

from .OnnxDet import OnnxDetector as _OnnxDetector
from .OnnxDetMutTD import OnnxInferencePool as _OnnxInferencePool, TaskStatus as _TaskStatus, TaskStatus
import cv2
import numpy
import gc


class DevFeature:
    EnableMultithreaded = False


class ST_Detector:
    """
    单线程检测器，适用于单线程环境
    """

    def __init__(self, model_path, names=None, conf_threshold=0.3, iou_threshold=0.5, use_gpu=True):
        self.detector = _OnnxDetector(model_path, names, conf_threshold, iou_threshold, use_gpu)

    def detect(self, image):
        return self.detector.detect(image)

    def warm_up(self):
        dummy_image = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        for i in range(5):
            self.detector.detect(dummy_image)
        print('Warm-up completed.')

    def __del__(self):
        """析构函数，释放检测器资源"""
        if hasattr(self, 'detector') and self.detector:
            del self.detector
            self.detector = None
            gc.collect()
            print("ST_Detector resources released.")
        else:
            print("ST_Detector already released or not initialized.")

class MT_Detector:
    """
    多线程检测器，适用于多线程环境
    """
    class SafeOnnxInference:
        """ONNX推理的安全包装类"""

        def __init__(self):
            """初始化安全包装类"""
            self.engine = None
            self.initialized = False

        def initialize(self, min_threads, max_threads, model_path):
            """安全初始化推理引擎"""
            try:
                self.engine = _OnnxInferencePool()
                if self.engine.create_pool(min_threads=min_threads,
                                           max_threads=max_threads,
                                           model_path=model_path):
                    self.initialized = True
                    print(f"成功初始化线程池，线程数: {min_threads}-{max_threads}")
                    return True
                else:
                    print(f"初始化失败: {self.engine.get_last_error()}")
                    return False
            except Exception as e:
                print(f"初始化过程中发生异常: {e}")
                traceback.print_exc()
                return False

        def submit_task(self, image, conf_threshold=0.3, iou_threshold=0.5):
            """提交任务"""
            if not self.initialized or not self.engine:
                return -1
            return self.engine.submit_task(image, conf_threshold, iou_threshold)

        def wait_for_task_completion(self, task_id, timeout=30.0):
            """等待任务完成"""
            if not self.initialized or not self.engine:
                return False
            return self.engine.wait_for_task_completion(task_id, timeout)

        def get_task_status(self, task_id):
            """获取任务状态"""
            if not self.initialized or not self.engine:
                return _TaskStatus.TASK_FAILED
            return self.engine.get_task_status(task_id)

        def get_inference_result(self, task_id):
            """获取推理结果"""
            if not self.initialized or not self.engine:
                return None
            return self.engine.get_inference_result(task_id)

        def run_inference(self, image, conf_threshold=0.3, iou_threshold=0.5):
            """运行推理并返回结果"""
            if not self.initialized or not self.engine:
                return {"error": "引擎未初始化"}

            try:
                task_id = self.engine.submit_task(image, conf_threshold, iou_threshold)
                if task_id <= 0:
                    return {"error": "任务提交失败"}

                if not self.engine.wait_for_task_completion(task_id, timeout=30.0):
                    return {"error": "任务执行超时"}

                return self.engine.get_inference_result(task_id)
            except Exception as e:
                return {"error": f"推理过程出错: {str(e)}"}

        def get_pool_status(self):
            """获取线程池状态"""
            if not self.initialized or not self.engine:
                return {"error": "引擎未初始化"}

            try:
                return self.engine.get_pool_status()
            except Exception as e:
                return {"error": f"获取状态时出错: {str(e)}"}

        def cleanup(self):
            """安全清理资源"""
            if not self.initialized or not self.engine:
                return

            print("执行安全清理...")
            # 先等待所有任务完成
            try:
                status = self.get_pool_status()
                if isinstance(status, dict) and "queue_size" in status:
                    if status["queue_size"] > 0 or status["active_threads"] > status["idle_threads"]:
                        print("等待任务完成...")
                        time.sleep(2)  # 等待一段时间
            except:
                pass

            # 将引擎置为None，让Python的垃圾回收来处理它
            # 不直接调用destroy_pool()以避免崩溃
            print("释放引擎资源")
            self.engine = None
            self.initialized = False

            # 强制垃圾回收
            gc.collect()

            print("清理完成")

    def __init__(self, model_path, names=None, conf_threshold=0.3, iou_threshold=0.5, workers=4):
        if not DevFeature.EnableMultithreaded:
            raise RuntimeError("多线程检测器功能未启用，如果需要启用，请设置 DevFeature.EnableMultithreaded = True")
        self.detector = self.SafeOnnxInference()
        self.names = names
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.detector.initialize(min_threads=workers, max_threads=workers, model_path=model_path)
        logging.warning('Multithreaded ONNX detector is Develop Func, not Stable.')
        logging.warning('Multithreaded ONNX detector is Develop Func, not Stable.')
        logging.warning('Multithreaded ONNX detector is Develop Func, not Stable.')

    def warm_up(self):
        dummy_image = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        for i in range(5):
            warm_up_id = self.detector.submit_task(dummy_image, conf_threshold=self.conf, iou_threshold=self.iou)
            if not self.detector.wait_for_task_completion(warm_up_id, timeout=3):
                continue
        print('Warm-up completed.')

    def submit_task(self, image):
        """
        提交检测任务
        :param image: 输入图像
        :return: 任务ID
        """
        return self.detector.submit_task(image, conf_threshold=self.conf, iou_threshold=self.iou)

    def wait_for_task_completion(self, task_id, timeout=3.0):
        """
        等待任务完成
        :param task_id: 任务ID
        :param timeout: 超时时间
        :return: 是否成功完成任务
        """
        return self.detector.wait_for_task_completion(task_id, timeout)

    def get_task_status(self, task_id):
        """

        :param task_id:
        :return:
        """
        if self.detector.get_task_status(task_id) == TaskStatus.TASK_COMPLETED:
            return True
        else:
            return False

    def get_result(self, task_id):
        """
        获取任务结果
        :param task_id: 任务ID
        :return: 检测结果
        """
        result = self.detector.get_inference_result(task_id)
        results = {}
        for i in self.names:
            results[str(i)] = []
        try:
            count = result['detection_count']
        except KeyError:
            logging.error("检测结果中未包含 'detection_count' 键，可能是推理未完成或推理失败。")
            return results
        if result['detection_count'] > 0 and 'detections' in result:
            for i in range(result['detection_count']):
                detection = result['detections'][i]
                cid = int(detection['class_id'])
                p1 = int(detection['box']['x'])
                p2 = int(detection['box']['y'])
                p3 = p1 + int(detection['box']['width'])
                p4 = p2 + int(detection['box']['height'])
                data = {
                    'confidence': round(detection['confidence'], 2),
                    'box': [(p1, p2), (p3, p4)],
                    'center': (int((p1 + p3) // 2), int((p2 + p4) // 2)),
                }
                results[str(self.names[cid])].append(data)
        return results

    def get_pool_status(self):
        """
        获取线程池状态
        :return: 线程池状态
        """
        return self.detector.get_pool_status()

    def cleanup(self):
        """
        清理资源
        """
        self.detector.cleanup()
        print("MT_Detector resources released.")