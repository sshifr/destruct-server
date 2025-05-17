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
        # Ensure all strings are properly escaped
        return json.dumps(data, ensure_ascii=False).replace('\n', '\\n').replace('\r', '\\r')
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"JSON serialization error: {str(e)}"
        })

class CameraAnalyzer:
    def __init__(self, model_path, camera_id=0, show_video=True):
        print(safe_json_dumps({"status": "info", "message": f"Инициализация с моделью: {model_path}, камера: {camera_id}"}))
        
        # Инициализируем атрибуты
        self.camera_id = camera_id
        self.show_video = show_video
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.processing_thread = None
        self.capture_thread = None
        
        # Создаем директорию для сохранения результатов
        self.save_dir = os.path.join('runs', 'detect', 'camera')
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

    def start(self):
        print(safe_json_dumps({"status": "info", "message": f"Попытка открыть камеру {self.camera_id}..."}))
        self.is_running = True
        
        # Пробуем открыть камеру несколько раз
        for attempt in range(3):
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if self.cap.isOpened():
                    # Устанавливаем RGB режим для камеры
                    self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    print(safe_json_dumps({"status": "info", "message": f"Камера успешно открыта с попытки {attempt + 1}"}))
                    break
                else:
                    print(safe_json_dumps({"status": "warning", "message": f"Попытка {attempt + 1} открыть камеру не удалась"}))
                    time.sleep(1)
            except Exception as e:
                print(safe_json_dumps({"status": "error", "message": f"Ошибка при открытии камеры: {str(e)}"}))
                time.sleep(1)
        
        if not self.cap.isOpened():
            print(safe_json_dumps({"status": "error", "message": "Не удалось открыть камеру после всех попыток"}))
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

    def _capture_frames(self):
        frame_count = 0
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(safe_json_dumps({"status": "error", "message": "Ошибка чтения кадра"}))
                    time.sleep(0.1)
                    continue

                # Проверяем формат кадра
                if frame is not None and frame.size > 0:
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
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
                                    if self.show_video:
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
                                # Проверяем, что кадр не пустой и имеет правильный формат
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
                            if dangerous_objects:
                                print(safe_json_dumps({
                                    "status": "warning",
                                    "message": f"Обнаружены потенциально опасные объекты: {', '.join([d['class'] for d in dangerous_objects])}"
                                }))
                                
                                # Сохраняем кадр с обнаруженными объектами
                                try:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    save_path = os.path.join(self.save_dir, f'detection_{timestamp}.jpg')
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
                time.sleep(0.1)  # Добавляем небольшую задержку при ошибке

def main():
    if len(sys.argv) < 2:
        print(safe_json_dumps({"status": "error", "message": "Необходимо указать путь к модели"}))
        return

    # Используем относительный путь от директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', sys.argv[1])
    camera_id = 0  # Используем первую камеру

    print(safe_json_dumps({
        "status": "info",
        "message": f"Запуск анализа с параметрами:\nМодель: {model_path}\nКамера: {camera_id}"
    }))

    analyzer = CameraAnalyzer(model_path, camera_id)
    
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