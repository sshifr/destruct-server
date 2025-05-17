import torch
from typing import List, Dict, Union, Optional
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import whisper
import string
import re
import argparse

def analyze_audio(filepath: str) -> str:
    # Анализ эмоций по аудио
    try:
        # Загрузка аудио
        y, sr = librosa.load(filepath)
        
        # Извлечение признаков
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Анализ признаков
        mfccs_mean = np.mean(mfccs, axis=1)
        spectral_centroids_mean = np.mean(spectral_centroids)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        rms_mean = np.mean(rms)
        pitch_mean = np.mean(pitch)
        
        # Определение эмоции на основе признаков
        # Злость: высокий спектральный центроид, высокая энергия, высокий zero crossing rate
        if (spectral_centroids_mean > 2500 and 
            rms_mean > 0.1 and 
            zero_crossing_rate_mean > 0.08):
            emotion = "angry"
        # Грусть: низкий спектральный центроид, низкая энергия, низкий zero crossing rate
        elif (spectral_centroids_mean < 1800 and 
              rms_mean < 0.05 and 
              zero_crossing_rate_mean < 0.04):
            emotion = "sad"
        # Счастье: высокий спектральный центроид, высокая энергия, средний zero crossing rate
        elif (spectral_centroids_mean > 2200 and 
              rms_mean > 0.08 and 
              0.04 < zero_crossing_rate_mean < 0.08):
            emotion = "happy"
        # Страх: высокий спектральный центроид, низкая энергия, высокий zero crossing rate
        elif (spectral_centroids_mean > 2300 and 
              rms_mean < 0.06 and 
              zero_crossing_rate_mean > 0.07):
            emotion = "fear"
        # Нейтральность: средние значения всех параметров
        else:
            emotion = "neutral"
            
        emotion_text = {
            "angry": "злость",
            "disgust": "Отвращение",
            "fear": "Страх",
            "happy": "Счастье",
            "sad": "Грусть",
            "neutral": "Нейтральность"
        }.get(emotion, emotion)
    except Exception as e:
        print(f"Ошибка при анализе эмоций: {e}")
        emotion_text = "Нейтральность"

    # Транскрибация
    model = whisper.load_model("small")
    result = model.transcribe(filepath)
    text = result["text"]

    # Поиск националистических слов
    nationalist_words_list: List[str] = []
    patterns: List[str] = [
        r"\bнацис\w+|\b\w+нацис\w+",
        r"\bнационалис\w+|\b\w+националис\w+",
        r"\bсверг\w+|\b\w+сверг\w+",
        r"\bсверж\w+|\b\w+сверж\w+",
        r"\bхач\w+|\b\w+хач\w+",
        r"\bчурк\w+|\b\w+чурк\w+",
        r"\bузкоглаз\w+|\b\w+узкоглаз\w+",
        r"\bпиздоглаз\w+|\b\w+пиздоглаз\w+",
        r"\bчерножоп\w+|\b\w+черножоп\w+",
        r"\bчёрножоп\w+|\b\w+чёрножоп\w+",
        r"\bрузг\w+|\b\w+рузг\w+",
        r"\bмоскал\w+|\b\w+москал\w+",
        r"\bватник\w+|\b\w+ватник\w+",
        r"\bкарсак\w+|\b\w+карсак\w+"
    ]

    for pattern in patterns:
        nationalist_words_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Поиск упоминаний ВС РФ
    vs_rf_list: List[str] = []
    vs_rf_patterns: List[str] = [
        r"\bвсрф\b|\bвс рф\b|\bв срф\b|\bвср ф\b|\bwsrf\b|\bw srf\b|\bws rf\b|\bwsr f\b",
        r"\bвооруженные силы российской федерации\b|\bвооружённые силы российской федерации\b"
    ]
    for pattern in vs_rf_patterns:
        vs_rf_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Поиск террористических слов
    terror_words_list: List[str] = []
    terror_patterns: List[str] = [
        r"\bподорв\w+|\b\w+подорв\w+",
        r"\bзаложн\w+|\b\w+заложн\w+",
        r"\bрасстрел\w+|\b\w+расстрел\w+",
        r"\bобезглав\w+|\b\w+обезглав\w+",
        r"\bподрыв\w+|\b\w+подрыв\w+",
        r"\bджихад\w+|\b\w+джихад\w+"
    ]
    for pattern in terror_patterns:
        terror_words_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Поиск одобрительных слов
    approve_words_list: List[str] = []
    approve_patterns: List[str] = [
        r"\bвосстанов\w+|\b\w+восстанов\w+",
        r"\bхорош\w+|\b\w+хорош\w+",
        r"\bлучш\w+|\b\w+лучш\w+",
        r"\bодобр\w+|\b\w+одобр\w+"
    ]
    for pattern in approve_patterns:
        approve_words_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Поиск нецензурных слов
    sweat_words_list: List[str] = []
    sweat_patterns: List[str] = [
        r"\bеба\w+|\b\w+еба\w+",
        r"\bбля\w+|\b\w+бля\w+",
        r"\bёба\w+|\b\w+ёба\w+",
        r"\bпизд\w+|\b\w+пизд\w+",
        r"\bхуй\w+|\b\w+хуй\w+",
        r"\bхуя\w+|\b\w+хуя\w+",
        r"\bхуё\w+|\b\w+хуё\w+",
        r"\bхуе\w+|\b\w+хуе\w+",
        r"\bсука\w+|\b\w+сука\w+",
        r"\bсучь\w+|\b\w+сучь\w+",
        r"\bсуки\w+|\b\w+суки\w+",
        r"\bсучи\w+|\b\w+сучи\w+"
    ]
    for pattern in sweat_patterns:
        sweat_words_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Поиск нацистских слов
    nazi_words_list: List[str] = []
    nazi_patterns: List[str] = [
        r"\bгитлер\w+|\b\w+гитлер\w+",
        r"\bгеббельс\w+|\b\w+геббельс\w+",
        r"\bнацис\w+|\b\w+нацис\w+",
        r"\bнациз\w+|\b\w+нациз\w+",
        r"\bгеноцид\w+|\b\w+геноцид\w+"
    ]
    for pattern in nazi_patterns:
        nazi_words_list.extend(re.findall(pattern, text.lower().translate(str.maketrans('', '', string.punctuation)).replace('Ё', 'Е').replace('ё', 'е')))

    # Определение типа контента и рекомендаций
    content_type: str = ""
    recommendations: str = ""

    if emotion == "happy": 
        if nationalist_words_list:
            content_type = "«Разжигание ненависти»"
            recommendations = "Ответственность по УК РФ Статья 280. Публичные призывы к осуществлению экстремистской деятельности.\nОтветственность по УК РФ Статья 128.1. Клевета."
        elif terror_words_list:
            content_type = "«Упоминание терроризма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        elif sweat_words_list:
            content_type = "«Нецензурная лексика»"
            recommendations = "Обратить внимание на психологическое состояние человека.\nОтветственность по КоАП РФ Статья 20.1. Мелкое хулиганство."
        elif nazi_words_list and approve_words_list:
            content_type = "«Упоминание нацизма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        else:
            content_type = "Отсутствие деструктивного контента"
            recommendations = "Обратить внимание на психологическое состояние человека."

    elif emotion == "angry":
        if nationalist_words_list:
            content_type = "«Разжигание ненависти»"
            recommendations = "Обратить внимание на психологическое состояние человека.\nОтветственность по УК РФ Статья 280. Публичные призывы к осуществлению экстремистской деятельности."
        elif terror_words_list:
            content_type = "«Упоминание терроризма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        elif sweat_words_list:
            content_type = "«Нецензурная лексика»"
            recommendations = "Обратить внимание на психологическое состояние человека.\nОтветственность по КоАП РФ Статья 20.1. Мелкое хулиганство."
        elif nazi_words_list and approve_words_list:
            content_type = "«Упоминание нацизма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        else:
            content_type = "Отсутствие деструктивного контента"
            recommendations = "Обратить внимание на психологическое состояние человека."

    elif emotion in ["sad", "fear"]:
        if nationalist_words_list:
            content_type = "«Разжигание ненависти»"
            recommendations = "Обратить внимание на психологическое состояние человека.\nОтветственность по УК РФ Статья 280. Публичные призывы к осуществлению экстремистской деятельности.\nОтветственность по УК РФ Статья 206. Захват заложника"
        elif terror_words_list:
            content_type = "«Упоминание терроризма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        elif sweat_words_list:
            content_type = "«Нецензурная лексика»"
            recommendations = "Обратить внимание на психологическое состояние человека.\nОтветственность по КоАП РФ Статья 20.1. Мелкое хулиганство."
        elif nazi_words_list and approve_words_list:
            content_type = "«Упоминание нацизма»"
            recommendations = "Обратить внимание на психологическое состояние человека."
        else:
            content_type = "Отсутствие деструктивного контента"
            recommendations = "Обратить внимание на психологическое состояние человека."

    else:  # neutral
        if nationalist_words_list:
            content_type = "«Разжигание ненависти»"
            recommendations = "Ответственность по УК РФ Статья 280. Публичные призывы к осуществлению экстремистской деятельности."
        elif terror_words_list:
            content_type = "«Упоминание терроризма»"
            recommendations = "Не принимать какие-либо действия."
        elif sweat_words_list:
            content_type = "«Нецензурная лексика»"
            recommendations = "Ответственность по КоАП РФ Статья 20.1. Мелкое хулиганство."
        elif nazi_words_list and approve_words_list:
            content_type = "«Упоминание нацизма»"
            recommendations = "Не принимать какие-либо действия."
    else:
            content_type = "Отсутствие деструктивного контента"
            recommendations = "Не принимать какие-либо действия."

    # Формируем итоговый результат
    result = f"Паралингвистический признак аудиосообщения: {emotion_text}\n"
    result += f"Содержательная часть: {content_type}\n"
    result += f"Меры, рекомендуемые к принятию: {recommendations}"

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to audio file')
    args = parser.parse_args()

    result = analyze_audio(args.source)
    print(result)

if __name__ == '__main__':
    main()