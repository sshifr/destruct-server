import argparse
import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO
import sys

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # Очищаем директорию, если она существует
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

def is_night_mode(image):
    # Конвертируем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Вычисляем среднюю яркость
    brightness = np.mean(gray)
    # Если средняя яркость меньше порога, считаем что это ночь
    return brightness < 100

def detect_motion(frame1, frame2, threshold=30):
    if frame1 is None or frame2 is None:
        return False
    
    # Конвертируем кадры в оттенки серого
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Вычисляем разницу между кадрами
    diff = cv2.absdiff(gray1, gray2)
    
    # Применяем пороговое значение
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Находим контуры движения
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Если есть достаточно большие контуры, считаем что есть движение
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Порог площади контура
            return True
    
    return False

def process_results(results, names):
    detected_classes = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            detected_classes.add(names[cls])
    
    if detected_classes:
        print(f"detected {len(detected_classes)} objects: {', '.join(detected_classes)}")
        
        # Проверяем на опасные объекты
        dangerous_objects = {
            'antifa', 'cocaine', 'confederate-flag', 'destroy',
            'fire', 'glass-defect', 'gun', 'heroin', 'isis',
            'knife', 'marijuana', 'rocket', 'shrooms', 'smoke', 'swastika',
            'wolfsangel', 'celtic_cross', 'Violence', 'cigarette', 'graffiti'
        }
        
        found_dangerous = [obj for obj in detected_classes if obj.lower() in dangerous_objects]
        if found_dangerous:
            print(f"WARNING: Dangerous objects detected: {', '.join(found_dangerous)}")
            return detected_classes, True, found_dangerous  # Возвращаем множество классов, флаг опасности и список опасных объектов
            
    return detected_classes, False, []  # Возвращаем множество классов, флаг опасности и пустой список опасных объектов

def save_danger_frame(frame, output_dir, source_path):
    """Сохраняет кадр с опасным объектом"""
    try:
        # Создаем имя файла на основе исходного файла
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        frame_filename = f"{base_name}_danger.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Убеждаемся, что директория существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем кадр
        print(f"Attempting to save frame to: {frame_path}")
        success = cv2.imwrite(frame_path, frame)
        if success:
            print(f"Successfully saved dangerous frame to: {frame_path}")
            return frame_path
        else:
            print(f"Failed to save frame to: {frame_path}")
            return None
    except Exception as e:
        print(f"Error saving dangerous frame: {e}")
        return None

def process_frame(frame, model, device, names, motion_detection=False, night_mode=False):
    # Проверяем ночной режим если включен
    if night_mode:
        if is_night_mode(frame):
            print("Night mode detected: Night scene")
        else:
            print("Night mode detected: Day scene")

    # Проверяем движение если включен датчик движения
    if motion_detection:
        # Здесь должна быть логика определения движения
        # Например, сравнение с предыдущим кадром
        print("Motion detected: Processing frame for motion detection")

    # Остальная логика обработки кадра
    results = model(frame)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save', action='store_true', help='Save results to video/image')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated class names to include')
    parser.add_argument('--show', action='store_true', help='Show detection window')
    parser.add_argument('--project', type=str, default='runs/detect', help='Save results to project/name')
    parser.add_argument('--name', type=str, default='predict', help='Save results to project/name')
    parser.add_argument('--motion-detection', action='store_true', help='Enable motion detection')
    parser.add_argument('--night-mode', action='store_true', help='Enable night mode detection')
    parser.add_argument('--quick-search', action='store_true', help='Stop processing when dangerous object is detected')
    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)
    print(f"Loaded model: {args.weights}")
    
    # Parse classes if provided
    if args.classes:
        try:
            class_names = [x.strip() for x in args.classes.split(',')]
            print(f"Using classes: {class_names}")
            
            all_classes = model.names
            class_indices = []
            for name in class_names:
                for idx, class_name in all_classes.items():
                    if class_name == name:
                        class_indices.append(idx)
                        break
            
            print(f"Found class indices: {class_indices}")
            classes = class_indices
        except Exception as e:
            print(f"Error parsing classes: {e}")
            classes = None
    else:
        classes = None

    # Create output directory
    output_dir = os.path.join(args.project, args.name)
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")

    # Проверяем, является ли источник изображением или видео
    is_image = args.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    
    if is_image:
        # Обработка изображения
        print(f"Processing image: {args.source}")
        
        # Запуск YOLO с сохранением
        results = model.predict(
            source=args.source,
            conf=args.conf,
            save=True,
            save_txt=args.save_txt,
            classes=classes,
            project=args.project,
            name=args.name,
            exist_ok=True,
            show=args.show
        )
        
        # Обрабатываем результаты
        detected_classes, has_dangerous, dangerous_objects = process_results(results, model.names)
        if args.quick_search and has_dangerous:
            print("Quick search mode: Dangerous object detected, stopping processing")
            print(f"Found dangerous objects: {', '.join(dangerous_objects)}")
            
            # Сохраняем изображение с опасным объектом
            frame = cv2.imread(args.source)
            if frame is not None:
                save_danger_frame(frame, output_dir, args.source)
            
            # Принудительно завершаем процесс
            print("Exiting process due to dangerous object detection")
            os._exit(0)
        
        # Проверяем ночной режим для изображения
        if args.night_mode:
            frame = cv2.imread(args.source)
            if is_night_mode(frame):
                print("Night mode detected: Night scene")
            else:
                print("Night mode detected: Day scene")
        
        print(f"Image processing completed. Results saved to: {output_dir}")

    else:
        # Обработка видео
        print(f"Processing video: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error opening video source: {args.source}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        if args.save:
            output_filename = os.path.basename(args.source)
            output_path = os.path.join(output_dir, output_filename)
            print(f"Saving to: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize motion detection variables
        prev_frame = None
        frame_count = 0
        all_detected_classes = set()
        dangerous_detected = False

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")

                # Проверяем ночной режим
                if args.night_mode:
                    if is_night_mode(frame):
                        print("Night mode detected: Night scene")
                    else:
                        print("Night mode detected: Day scene")

                # Проверяем движение
                if args.motion_detection:
                    if prev_frame is not None and detect_motion(prev_frame, frame):
                        print("Motion detected")
                    prev_frame = frame.copy()

                # Run YOLO detection
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    save_txt=args.save_txt,
                    classes=classes,
                    stream=True,
                    exist_ok=True
                )

                # Process results
                for result in results:
                    # Получаем изображение с боксами
                    annotated_frame = result.plot()
                    
                    # Обрабатываем результаты
                    frame_classes, has_dangerous, dangerous_objects = process_results([result], model.names)
                    if args.quick_search and has_dangerous:
                        dangerous_detected = True
                        print("Quick search mode: Dangerous object detected, stopping processing")
                        print(f"Found dangerous objects: {', '.join(dangerous_objects)}")
                        
                        # Сохраняем кадр с опасным объектом
                        save_danger_frame(annotated_frame, output_dir, args.source)
                        
                        # Сохраняем текущий кадр в видео если включено сохранение
                        if args.save:
                            out.write(annotated_frame)
                        
                        # Закрываем все ресурсы
                        cap.release()
                        if args.save:
                            out.release()
                        if args.show:
                            cv2.destroyAllWindows()
                        
                        # Принудительно завершаем процесс
                        print("Exiting process due to dangerous object detection")
                        os._exit(0)
                    
                    all_detected_classes.update(frame_classes)
                    
                    if args.save:
                        out.write(annotated_frame)
                    
                    if args.show:
                        cv2.imshow('Detection', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                if dangerous_detected:
                    break

        finally:
            # Cleanup
            cap.release()
            if args.save:
                out.release()
            if args.show:
                cv2.destroyAllWindows()

        # Выводим итоговый список обнаруженных классов
        if all_detected_classes:
            print(f"Final list of detected objects: {', '.join(all_detected_classes)}")

        print(f"Video processing completed. Results saved to: {output_dir}")

if __name__ == '__main__':
    main() 