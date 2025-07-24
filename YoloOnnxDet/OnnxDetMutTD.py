import ctypes
import numpy as np
import cv2
from ctypes import Structure, c_int, c_float, c_double, c_char_p, c_bool, POINTER, c_ubyte, c_char
from enum import IntEnum
import logging
import time
from importlib.resources import files as resource_path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ONNXInference")


class TaskStatus(IntEnum):
    """任务状态枚举"""
    TASK_WAITING = 0
    TASK_PROCESSING = 1
    TASK_COMPLETED = 2
    TASK_FAILED = 3


class PoolStatus(Structure):
    """线程池状态结构体"""
    _fields_ = [
        ("active_threads", c_int),
        ("idle_threads", c_int),
        ("queue_size", c_int),
        ("total_tasks", c_int),
        ("failed_tasks", c_int)
    ]


class DetectionBox(Structure):
    """检测框结构体 - 对应cv::Rect"""
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int)
    ]


class DetectionResult(Structure):
    """检测结果结构体"""
    _fields_ = [
        ("box", DetectionBox),
        ("confidence", c_float),
        ("class_id", c_int)
    ]


class InputData(Structure):
    """输入数据结构体"""
    _fields_ = [
        ("image_data", POINTER(c_ubyte)),
        ("image_width", c_int),
        ("image_height", c_int),
        ("image_channels", c_int),
        ("conf_threshold", c_double),
        ("iou_threshold", c_double)
    ]


class OutputData(Structure):
    """输出数据结构体 - 修复版本"""
    _fields_ = [
        ("detections", POINTER(DetectionResult)),
        ("detection_count", c_int),
        ("inference_time", c_float),
        ("error_message", c_char * 256)  # 固定长度数组
    ]


class OnnxInferencePool:
    """ONNX推理线程池Python封装类 - 修复版本"""

    def __init__(self):
        self.dll = None
        dll_path = str(resource_path('YoloOnnxDet').joinpath('OnnxMutTD.dll'))
        self.is_initialized = False
        self._image_buffers = []  # 保存图像数据引用
        self._result_buffers = []  # 保存结果数据引用
        self._load_dll(dll_path)
        self._setup_function_signatures()

    def _load_dll(self, dll_path: str):
        """加载DLL库"""
        try:
            self.dll = ctypes.CDLL(dll_path)
            # logger.info(f"Successfully loaded DLL: {dll_path}")
        except Exception as e:
            logger.error(f"Failed to load DLL {dll_path}: {e}")
            raise RuntimeError(f"Cannot load DLL: {e}")

    def _setup_function_signatures(self):
        """设置函数签名"""
        try:
            # CreateInferencePool
            self.dll.CreateInferencePool.argtypes = [c_int, c_int, c_char_p]
            self.dll.CreateInferencePool.restype = c_bool

            # DestroyInferencePool
            self.dll.DestroyInferencePool.argtypes = []
            self.dll.DestroyInferencePool.restype = None

            # GetPoolStatus
            self.dll.GetPoolStatus.argtypes = []
            self.dll.GetPoolStatus.restype = PoolStatus

            # SubmitInferenceTask
            self.dll.SubmitInferenceTask.argtypes = [POINTER(InputData)]
            self.dll.SubmitInferenceTask.restype = c_int

            # GetTaskStatus
            self.dll.GetTaskStatus.argtypes = [c_int]
            self.dll.GetTaskStatus.restype = c_int

            # GetInferenceResult
            self.dll.GetInferenceResult.argtypes = [c_int, POINTER(OutputData)]
            self.dll.GetInferenceResult.restype = c_bool

            # CancelTask
            self.dll.CancelTask.argtypes = [c_int]
            self.dll.CancelTask.restype = c_bool

            # GetLastError
            self.dll.GetLastError.argtypes = []
            self.dll.GetLastError.restype = c_char_p

        except Exception as e:
            logger.error(f"Failed to setup function signatures: {e}")
            raise

    def create_pool(self, min_threads: int, max_threads: int, model_path: str) -> bool:
        """创建推理线程池"""
        try:
            if self.is_initialized:
                logger.warning("Pool already initialized")
                return True

            model_path_bytes = model_path.encode('utf-8')
            result = self.dll.CreateInferencePool(min_threads, max_threads, model_path_bytes)
            self.is_initialized = result

            if result:
                logger.info(f"Inference pool created successfully with {min_threads}-{max_threads} threads")
            else:
                error_msg = self.get_last_error()
                logger.error(f"Failed to create inference pool: {error_msg}")
            return result
        except Exception as e:
            logger.error(f"Exception in create_pool: {e}")
            return False

    def destroy_pool(self):
        """销毁推理线程池"""
        try:
            if self.is_initialized:
                # 清理缓冲区
                self._cleanup_buffers()

                # 销毁线程池
                self.dll.DestroyInferencePool()
                self.is_initialized = False
                logger.info("Inference pool destroyed")
        except Exception as e:
            logger.error(f"Exception in destroy_pool: {e}")
            # 即使出错也要标记为未初始化
            self.is_initialized = False

    def _cleanup_buffers(self):
        """清理所有缓冲区"""
        try:
            # 清理结果缓冲区
            for result_ptr in self._result_buffers:
                if result_ptr:
                    try:
                        # 释放C++分配的内存
                        ctypes.windll.msvcrt.free(result_ptr)
                    except:
                        pass
            self._result_buffers.clear()

            # 清理图像缓冲区
            self._image_buffers.clear()

        except Exception as e:
            logger.error(f"Exception in cleanup_buffers: {e}")

    def get_pool_status(self) -> dict:
        """获取线程池状态"""
        if not self.is_initialized:
            return {
                'active_threads': 0,
                'idle_threads': 0,
                'queue_size': 0,
                'total_tasks': 0,
                'failed_tasks': 0
            }

        try:
            status = self.dll.GetPoolStatus()
            return {
                'active_threads': status.active_threads,
                'idle_threads': status.idle_threads,
                'queue_size': status.queue_size,
                'total_tasks': status.total_tasks,
                'failed_tasks': status.failed_tasks
            }
        except Exception as e:
            logger.error(f"Exception in get_pool_status: {e}")
            return {}

    def submit_task(self, image: np.ndarray, conf_threshold: float = 0.3,
                    iou_threshold: float = 0.5) -> int:
        """提交推理任务"""
        if not self.is_initialized:
            logger.error("Inference pool not initialized")
            return -1

        try:
            # 验证输入图像
            if image is None or image.size == 0:
                logger.error("Invalid image: empty or None")
                return -1

            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image shape: {image.shape}, expected (H, W, 3)")
                return -1

            # 确保图像是连续内存布局且为uint8类型
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)

            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # 创建图像副本并保存引用
            image_buffer = image.copy()
            self._image_buffers.append(image_buffer)

            # 创建输入数据结构
            input_data = InputData()
            input_data.image_data = image_buffer.ctypes.data_as(POINTER(c_ubyte))
            input_data.image_height = image_buffer.shape[0]
            input_data.image_width = image_buffer.shape[1]
            input_data.image_channels = image_buffer.shape[2]
            input_data.conf_threshold = conf_threshold
            input_data.iou_threshold = iou_threshold

            # 提交任务
            task_id = self.dll.SubmitInferenceTask(ctypes.byref(input_data))

            if task_id > 0:
                # logger.info(f"Task submitted successfully with ID: {task_id}")
                pass
            else:
                error_msg = self.get_last_error()
                logger.error(f"Failed to submit task: {error_msg}")
                # 移除失败的图像缓冲区
                if self._image_buffers:
                    self._image_buffers.pop()

            return task_id

        except Exception as e:
            logger.error(f"Exception in submit_task: {e}")
            return -1

    def get_task_status(self, task_id: int) -> TaskStatus:
        """获取任务状态"""
        if not self.is_initialized:
            return TaskStatus.TASK_FAILED

        try:
            status = self.dll.GetTaskStatus(task_id)
            return TaskStatus(status)
        except Exception as e:
            logger.error(f"Exception in get_task_status: {e}")
            return TaskStatus.TASK_FAILED

    def get_inference_result(self, task_id: int) -> dict:
        """获取推理结果 - 修复版本"""
        if not self.is_initialized:
            return {}

        try:
            output_data = OutputData()

            # 初始化输出数据
            output_data.detections = None
            output_data.detection_count = 0
            output_data.inference_time = 0.0

            success = self.dll.GetInferenceResult(task_id, ctypes.byref(output_data))

            if not success:
                error_msg = self.get_last_error()
                logger.error(f"Failed to get inference result for task {task_id}: {error_msg}")
                return {}

            # 安全地转换检测结果
            detections = []
            if output_data.detection_count > 0 and output_data.detections:
                try:
                    # 保存结果指针用于后续清理
                    self._result_buffers.append(output_data.detections)

                    for i in range(output_data.detection_count):
                        detection = output_data.detections[i]
                        detections.append({
                            'box': {
                                'x': detection.box.x,
                                'y': detection.box.y,
                                'width': detection.box.width,
                                'height': detection.box.height
                            },
                            'confidence': float(detection.confidence),
                            'class_id': int(detection.class_id)
                        })

                except Exception as e:
                    logger.error(f"Error processing detection results: {e}")
                    detections = []

            # 处理错误信息
            error_message = ""
            try:
                if output_data.error_message:
                    error_message = output_data.error_message.decode('utf-8', errors='ignore').strip()
            except:
                error_message = ""

            result = {
                'detections': detections,
                'detection_count': output_data.detection_count,
                'inference_time': float(output_data.inference_time),
                'error_message': error_message
            }

            return result

        except Exception as e:
            logger.error(f"Exception in get_inference_result: {e}")
            return {}

    def cancel_task(self, task_id: int) -> bool:
        """取消任务"""
        if not self.is_initialized:
            return False

        try:
            return bool(self.dll.CancelTask(task_id))
        except Exception as e:
            logger.error(f"Exception in cancel_task: {e}")
            return False

    def get_last_error(self) -> str:
        """获取最后的错误信息"""
        try:
            if not self.dll:
                return "DLL not loaded"

            error_ptr = self.dll.GetLastError()
            if error_ptr:
                return error_ptr.decode('utf-8', errors='ignore')
            return ""
        except Exception as e:
            logger.error(f"Exception in get_last_error: {e}")
            return str(e)

    def wait_for_task_completion(self, task_id: int, timeout: float = 30.0) -> bool:
        """等待任务完成"""
        start_time = time.time()
        check_interval = 0.1  # 检查间隔100ms

        while time.time() - start_time < timeout:
            try:
                status = self.get_task_status(task_id)
                if status == TaskStatus.TASK_COMPLETED:
                    return True
                elif status == TaskStatus.TASK_FAILED:
                    logger.warning(f"Task {task_id} failed")
                    return False

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Exception while waiting for task {task_id}: {e}")
                return False

        logger.warning(f"Task {task_id} timeout after {timeout} seconds")
        return False

    def process_image_sync(self, image: np.ndarray, conf_threshold: float = 0.3,
                           iou_threshold: float = 0.5, timeout: float = 30.0) -> dict:
        """同步处理图像的便捷方法"""
        task_id = self.submit_task(image, conf_threshold, iou_threshold)
        if task_id <= 0:
            return {}

        if self.wait_for_task_completion(task_id, timeout):
            return self.get_inference_result(task_id)
        else:
            # 尝试取消超时的任务
            self.cancel_task(task_id)
            return {}

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy_pool()