import argparse
import cv2
import numpy as np
import os
import shutil
from deepface import DeepFace
from mtcnn import MTCNN
import json

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

def process_emotions(frame, detector):
    try:
        # Находим лица на кадре
        faces = detector.detect_faces(frame)
        
        emotions_data = []
        for face in faces:
            x, y, w, h = face['box']
            # Увеличиваем область лица для лучшего распознавания эмоций
            x = max(0, x - int(w * 0.1))
            y = max(0, y - int(h * 0.1))
            w = int(w * 1.2)
            h = int(h * 1.2)
            
            # Обрезаем лицо
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
                
            # Анализируем эмоции
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Получаем эмоции
            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            
            # Добавляем информацию о лице и эмоциях
            emotions_data.append({
                'box': [x, y, w, h],
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            })
            
            # Рисуем рамку и подпись
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Добавляем фон для текста
            text = dominant_emotion
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0  # Значительно увеличили размер шрифта
            thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Рисуем фон для текста
            cv2.rectangle(frame, 
                        (x, y - text_height - 20), 
                        (x + text_width + 20, y), 
                        (0, 0, 0), 
                        -1)
            
            # Рисуем текст
            cv2.putText(frame, 
                       text, 
                       (x + 10, y - 10),
                       font, 
                       font_scale, 
                       (0, 255, 0),
                       thickness)
            
            # Выводим информацию в логи
            print(f"\nОбнаружено лицо с эмоциями:")
            print(f"Доминирующая эмоция: {dominant_emotion}")
            print("Детальные оценки эмоций:")
            for emotion, score in emotions.items():
                print(f"- {emotion}: {score:.2f}%")
            
        return frame, emotions_data
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return frame, []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--save', action='store_true', help='Save results to video/image')
    parser.add_argument('--show', action='store_true', help='Show detection window')
    parser.add_argument('--project', type=str, default='../runs/detect', help='Save results to project/name')
    parser.add_argument('--name', type=str, default='emotions', help='Save results to project/name')
    args = parser.parse_args()

    # Инициализируем детектор лиц
    detector = MTCNN()
    
    # Создаем выходную директорию
    output_dir = os.path.join(args.project, args.name)
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")

    # Проверяем, является ли источник изображением или видео
    is_image = args.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    
    if is_image:
        # Обработка изображения
        print(f"\nОбработка изображения: {args.source}")
        frame = cv2.imread(args.source)
        
        # Обрабатываем эмоции
        processed_frame, emotions_data = process_emotions(frame, detector)
        
        # Сохраняем результаты
        if args.save:
            output_path = os.path.join(output_dir, os.path.basename(args.source))
            cv2.imwrite(output_path, processed_frame)
            
            # Сохраняем данные об эмоциях в JSON
            emotions_path = os.path.join(output_dir, 'emotions.json')
            with open(emotions_path, 'w') as f:
                json.dump(emotions_data, f, indent=4)
            
            print(f"\nРезультаты сохранены:")
            print(f"- Изображение: {output_path}")
            print(f"- Данные об эмоциях: {emotions_path}")
        
        # Показываем результат
        if args.show:
            cv2.imshow('Emotion Detection', processed_frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
            
    else:
        # Обработка видео
        print(f"\nОбработка видео: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error opening video source: {args.source}")
            return

        # Получаем свойства видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Инициализируем видеозапись
        if args.save:
            output_filename = os.path.basename(args.source)
            output_path = os.path.join(output_dir, output_filename)
            print(f"Saving to: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        all_emotions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"\nОбработка кадра {frame_count}/{total_frames}")

            # Обрабатываем эмоции
            processed_frame, emotions_data = process_emotions(frame, detector)
            all_emotions.extend(emotions_data)
            
            if args.save:
                out.write(processed_frame)
            
            if args.show:
                cv2.imshow('Emotion Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Сохраняем все эмоции в JSON
        if args.save:
            emotions_path = os.path.join(output_dir, 'emotions.json')
            with open(emotions_path, 'w') as f:
                json.dump(all_emotions, f, indent=4)
            
            print(f"\nРезультаты сохранены:")
            print(f"- Видео: {output_path}")
            print(f"- Данные об эмоциях: {emotions_path}")

        # Очистка
        cap.release()
        if args.save:
            out.release()
        if args.show:
            cv2.destroyAllWindows()

        print(f"\nОбработка видео завершена. Результаты сохранены в: {output_dir}")

if __name__ == '__main__':
    main() 