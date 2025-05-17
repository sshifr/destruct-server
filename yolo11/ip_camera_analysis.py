import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import json
import sys
import signal
import threading
from queue import Queue
import os
from datetime import datetime
import base64

def safe_json_dumps(data):
    try:
        return json.dumps(data, ensure_ascii=False).replace('\n', '\\n').replace('\r', '\\r')
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"JSON serialization error: {str(e)}"
        })

class IPCameraAnalyzer:
    def __init__(self, model_path, rtsp_url, motion_detection=False, night_mode=False):
        print(safe_json_dumps({
            "status": "info", 
            "message": f"Инициализация с моделью: {model_path}, RTSP URL: {rtsp_url}"
        }))
        
        # Инициализируем атрибуты
        self.rtsp_url = rtsp_url
        self.motion_detection = motion_detection
        self.night_mode = night_mode
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.processing_thread = None
        self.capture_thread = None
        
        # Создаем директорию для сохранения результатов
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(os.path.dirname(script_dir), 'runs', 'detect', 'ip_camera')
        os.makedirs(self.save_dir, exist_ok=True)
        print(safe_json_dumps({"status": "info", "message": f"Директория для сохранения: {self.save_dir}"}))

        # Проверяем существование модели
        if not os.path.exists(model_path):
            print(safe_json_dumps({
                "status": "error",
                "message": f"Модель не найдена: {model_path}"
            }))
            return

        try:
            print(safe_json_dumps({"status": "info", "message": "Загрузка модели YOLO..."}))
            self.model = YOLO(model_path)
            print(safe_json_dumps({"status": "info", "message": "Модель успешно загружена"}))
        except Exception as e:
            print(safe_json_dumps({
                "status": "error",
                "message": f"Ошибка загрузки модели: {str(e)}"
            }))
            return

        # Инициализируем детектор движения
        if self.motion_detection:
            self.motion_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
            self.prev_frame = None
            self.motion_threshold = 500  # Порог для определения движения

    def start(self):
        print(safe_json_dumps({"status": "info", "message": f"Попытка подключиться к RTSP потоку: {self.rtsp_url}"}))
        self.is_running = True
        
        # Пробуем открыть RTSP поток несколько раз
        for attempt in range(3):
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if self.cap.isOpened():
                    print(safe_json_dumps({"status": "info", "message": f"RTSP поток успешно открыт с попытки {attempt + 1}"}))
                    break
                else:
                    print(safe_json_dumps({"status": "warning", "message": f"Попытка {attempt + 1} открыть RTSP поток не удалась"}))
                    time.sleep(1)
            except Exception as e:
                print(safe_json_dumps({"status": "error", "message": f"Ошибка при открытии RTSP потока: {str(e)}"}))
                time.sleep(1)
        
        if not self.cap.isOpened():
            print(safe_json_dumps({"status": "error", "message": "Не удалось открыть RTSP поток после всех попыток"}))
            return False

        # Запускаем потоки для захвата и обработки кадров
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.processing_thread = threading.Thread(target=self._process_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        print(safe_json_dumps({"status": "info", "message": "Потоки обработки запущены"}))
        return True

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Очищаем очереди
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()

    def _detect_motion(self, frame):
        if not self.motion_detection:
            return False

        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Если это первый кадр, сохраняем его
        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        # Вычисляем разницу между текущим и предыдущим кадром
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Находим контуры
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Проверяем, есть ли значительное движение
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                motion_detected = True
                break

        # Обновляем предыдущий кадр
        self.prev_frame = gray

        return motion_detected

    def _is_night_mode(self, frame):
        if not self.night_mode:
            return False

        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Вычисляем среднюю яркость
        avg_brightness = np.mean(gray)
        
        # Если средняя яркость ниже порога, считаем что это ночной режим
        is_night = avg_brightness < 50  # Порог можно настроить
        
        # Всегда выводим информацию о типе сцены
        scene_type = "ночная" if is_night else "дневная"
        print(safe_json_dumps({
            "status": "info",
            "message": f"Текущая сцена: {scene_type} (яркость: {avg_brightness:.2f})"
        }))
        
        return is_night

    def _capture_frames(self):
        frame_count = 0
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(safe_json_dumps({"status": "error", "message": "Ошибка чтения кадра"}))
                    time.sleep(0.1)
                    continue

                # Проверяем и конвертируем в RGB для отображения
                if frame is not None and frame.size > 0:
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        print(safe_json_dumps({"status": "error", "message": "Неправильный формат кадра"}))
                        continue

                frame_count += 1
                if frame_count % 30 == 0:  # Логируем каждые 30 кадров
                    print(safe_json_dumps({"status": "info", "message": f"Обработано кадров: {frame_count}"}))

                # Очищаем очередь, если она полная
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass

                self.frame_queue.put(frame)
            except Exception as e:
                print(safe_json_dumps({"status": "error", "message": f"Ошибка в потоке захвата: {str(e)}"}))
                time.sleep(0.1)

    def _process_frames(self):
        frame_count = 0
        while self.is_running:
            try:
                # Получаем кадр из очереди с таймаутом
                try:
                    frame = self.frame_queue.get(timeout=1)
                except:
                    continue

                if frame is None or frame.size == 0:
                    print(safe_json_dumps({"status": "error", "message": "Получен пустой кадр"}))
                    continue

                frame_count += 1
                
                # Засекаем время начала обработки
                start_time = time.time()
                
                try:
                    # Проверяем движение и ночной режим
                    motion_detected = self._detect_motion(frame)
                    is_night = self._is_night_mode(frame)

                    # Логируем каждые 30 кадров
                    if frame_count % 30 == 0:
                        scene_type = "ночная" if is_night else "дневная"
                        print(safe_json_dumps({
                            "status": "info",
                            "message": f"Текущая сцена: {scene_type}"
                        }))

                    # Запускаем модель
                    results = self.model(frame, verbose=False)
                    
                    # Обрабатываем результаты
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            # Получаем информацию об обнаруженных объектах
                            detections = []
                            for box in boxes:
                                try:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    name = result.names[cls]
                                    detections.append({
                                        "class": name,
                                        "confidence": conf
                                    })
                                    
                                    # Рисуем рамку и подпись
                                    try:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        # Проверяем, что координаты находятся в пределах изображения
                                        h, w = frame.shape[:2]
                                        x1 = max(0, min(x1, w-1))
                                        y1 = max(0, min(y1, h-1))
                                        x2 = max(0, min(x2, w-1))
                                        y2 = max(0, min(y2, h-1))
                                        
                                        # Рисуем рамку
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        
                                        # Подготавливаем текст
                                        text = f"{name} {conf:.2f}"
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 0.5
                                        thickness = 2
                                        
                                        # Получаем размеры текста
                                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                        
                                        # Рисуем фон для текста
                                        cv2.rectangle(frame, 
                                                    (x1, y1 - text_height - 10),
                                                    (x1 + text_width, y1),
                                                    (0, 255, 0),
                                                    -1)
                                        
                                        # Рисуем текст
                                        cv2.putText(frame,
                                                  text,
                                                  (x1, y1 - 5),
                                                  font,
                                                  font_scale,
                                                  (0, 0, 0),
                                                  thickness)
                                    except Exception as e:
                                        print(safe_json_dumps({
                                            "status": "error",
                                            "message": f"Ошибка отрисовки бокса: {str(e)}"
                                        }))
                                        continue
                                except Exception as e:
                                    print(safe_json_dumps({
                                        "status": "error",
                                        "message": f"Ошибка обработки бокса: {str(e)}"
                                    }))
                                    continue
                            
                            # Отправляем результаты
                            print(safe_json_dumps({
                                "status": "info",
                                "message": f"Обнаружено {len(detections)} объектов: {', '.join([d['class'] for d in detections])}"
                            }))
                            
                            # Отправляем кадр с боксами в base64
                            try:
                                if frame is not None and frame.size > 0:
                                    # Создаем копию кадра для обработки
                                    frame_copy = frame.copy()
                                    
                                    # Устанавливаем параметры сжатия JPEG
                                    encode_params = [
                                        cv2.IMWRITE_JPEG_QUALITY, 95,
                                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                                    ]
                                    
                                    # Кодируем изображение
                                    success, jpeg = cv2.imencode('.jpg', frame_copy, encode_params)
                                    
                                    if success:
                                        b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                                        print(safe_json_dumps({
                                            "status": "frame",
                                            "image": b64
                                        }))
                                    else:
                                        print(safe_json_dumps({
                                            "status": "error",
                                            "message": "Ошибка кодирования JPEG"
                                        }))
                            except Exception as e:
                                print(safe_json_dumps({
                                    "status": "error",
                                    "message": f"Ошибка кодирования кадра: {str(e)}"
                                }))
                            
                            # Если обнаружены опасные объекты
                            dangerous_objects = [d for d in detections if d['class'] in {
                                'antifa', 'cocaine', 'confederate-flag', 'destroy',
                                'fire', 'glass-defect', 'gun', 'heroin', 'isis',
                                'knife', 'marijuana', 'rocket', 'shrooms', 'smoke', 'swastika',
                                'wolfsangel', 'celtic_cross', 'Violence', 'graffiti'
                            }]
                            
                            # Сохраняем кадр если:
                            # 1. Обнаружены опасные объекты
                            # 2. Обнаружено движение в ночном режиме
                            should_save = False
                            save_reason = []

                            if dangerous_objects:
                                should_save = True
                                save_reason.append("опасные объекты")
                                print(safe_json_dumps({
                                    "status": "warning",
                                    "message": f"Обнаружены потенциально опасные объекты: {', '.join([d['class'] for d in dangerous_objects])}"
                                }))

                            if motion_detected and is_night:
                                should_save = True
                                save_reason.append("движение в ночном режиме")
                                print(safe_json_dumps({
                                    "status": "warning",
                                    "message": "Обнаружено движение в ночном режиме!"
                                }))

                            if should_save:
                                try:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    reason_str = "_".join(save_reason)
                                    save_path = os.path.join(self.save_dir, f'detection_{reason_str}_{timestamp}.jpg')
                                    cv2.imwrite(save_path, frame)
                                    
                                    print(safe_json_dumps({
                                        "status": "info",
                                        "message": f"Сохранен кадр с обнаруженными объектами: {save_path}"
                                    }))
                                except Exception as e:
                                    print(safe_json_dumps({
                                        "status": "error",
                                        "message": f"Ошибка сохранения кадра: {str(e)}"
                                    }))
                        else:
                            if frame_count % 30 == 0:  # Логируем каждые 30 кадров
                                print(safe_json_dumps({
                                    "status": "info",
                                    "message": "Объекты не обнаружены"
                                }))
                                
                                # Отправляем кадр даже если объекты не обнаружены
                                try:
                                    if frame is not None and frame.size > 0:
                                        # Создаем копию кадра для обработки
                                        frame_copy = frame.copy()
                                        
                                        # Устанавливаем параметры сжатия JPEG
                                        encode_params = [
                                            cv2.IMWRITE_JPEG_QUALITY, 95,
                                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                                        ]
                                        
                                        # Кодируем изображение
                                        success, jpeg = cv2.imencode('.jpg', frame_copy, encode_params)
                                        
                                        if success:
                                            b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                                            print(safe_json_dumps({
                                                "status": "frame",
                                                "image": b64
                                            }))
                                        else:
                                            print(safe_json_dumps({
                                                "status": "error",
                                                "message": "Ошибка кодирования JPEG"
                                            }))
                                except Exception as e:
                                    print(safe_json_dumps({
                                        "status": "error",
                                        "message": f"Ошибка кодирования кадра: {str(e)}"
                                    }))
                except Exception as e:
                    print(safe_json_dumps({
                        "status": "error",
                        "message": f"Ошибка обработки кадра: {str(e)}"
                    }))
                    continue
                
                # Вычисляем время обработки
                process_time = time.time() - start_time
                if frame_count % 30 == 0:  # Логируем каждые 30 кадров
                    print(safe_json_dumps({
                        "status": "info",
                        "message": f"Время обработки: {process_time:.2f} сек"
                    }))
                
            except Exception as e:
                print(safe_json_dumps({
                    "status": "error",
                    "message": f"Ошибка в основном цикле обработки: {str(e)}"
                }))
                time.sleep(0.1)

def main():
    if len(sys.argv) < 3:
        print(safe_json_dumps({"status": "error", "message": "Необходимо указать модель и RTSP URL"}))
        return

    # Получаем параметры
    model_name = sys.argv[1]
    rtsp_url = sys.argv[2]
    motion_detection = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
    night_mode = len(sys.argv) > 4 and sys.argv[4].lower() == 'true'

    print(safe_json_dumps({
        "status": "info",
        "message": f"Получены параметры:\nМодель: {model_name}\nRTSP URL: {rtsp_url}\nДатчик движения: {motion_detection}\nНочной режим: {night_mode}\nАргументы: {sys.argv}"
    }))

    # Формируем путь к модели
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', model_name)

    print(safe_json_dumps({
        "status": "info",
        "message": f"Запуск анализа с параметрами:\nМодель: {model_path}\nRTSP URL: {rtsp_url}\nДатчик движения: {motion_detection}\nНочной режим: {night_mode}"
    }))

    analyzer = IPCameraAnalyzer(model_path, rtsp_url, motion_detection, night_mode)
    
    def signal_handler(signum, frame):
        print(safe_json_dumps({"status": "info", "message": "Получен сигнал остановки"}))
        analyzer.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if analyzer.start():
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            analyzer.stop()
    else:
        print(safe_json_dumps({"status": "error", "message": "Не удалось запустить анализ"}))

if __name__ == "__main__":
    main() 