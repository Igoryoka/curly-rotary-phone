from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import moviepy.editor as mp
import sys
import os
import uuid
import joblib
import tempfile
import soundfile as sf

# Додавання шляху до системного шляху для імпорту модулів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Імпорт функцій та змінних з модуля emotion_predictor
from emotion_detector.emotion_predictor import extract_feature, predict_emotion, emotions

app = Flask(__name__)
CORS(app)  # Дозвіл всіх CORS-запитів

# Допустимі розширення файлів
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov'}

# Функція для перевірки допустимих розширень файлів
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Функція для конвертації відео в аудіо
def convert_video_to_audio(video_path, audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)

# Функція для подавлення шуму
def denoise_audio(y, sr):
    y_denoised = librosa.effects.preemphasis(y)
    return y_denoised

# Функція для поділу аудіо на сегменти
def split_audio(y, sr, segment_duration=3):
    segment_length = int(segment_duration * sr)
    return [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Перевірка наявності файлу у запиті
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # Перевірка наявності імені файлу
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Перевірка допустимого типу файлу
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        # Збереження файлу у тимчасову директорію
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(temp_dir, temp_filename)
        file.save(file_path)

        # Конвертація відео в аудіо, якщо файл є відео
        if file_extension in ['mp4', 'avi', 'mov']:
            audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.wav")
            convert_video_to_audio(file_path, audio_path)
            os.remove(file_path)
        else:
            audio_path = file_path

        # Завантаження аудіофайлу
        y, sr = librosa.load(audio_path, sr=None)
        # Подача шуму
        y_denoised = denoise_audio(y, sr)
        # Поділ аудіо на сегменти
        segments = split_audio(y_denoised, sr, segment_duration=3)

        # Отримання абсолютного шляху до файлу моделі
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'emotion_detector', 'SVM_emotion_clf_24.pkl'))

        # Перевірка наявності файлу моделі
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found at {model_path}'}), 500

        segment_emotions = []
        # Аналіз кожного сегменту
        for segment in segments:
            if len(segment) == 0:
                continue
            segment_temp_path = os.path.join(temp_dir, f"segment_{uuid.uuid4()}.wav")
            sf.write(segment_temp_path, segment, sr)

            # Виділення ознак та передбачення емоції
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(segment_temp_path)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([ext_features])
            prediction = predict_emotion(model_path, features)
            emotion = emotions[prediction[0]]
            segment_emotions.append(emotion)

            os.remove(segment_temp_path)

        os.remove(audio_path)

        # Повернення передбачених емоцій
        return jsonify({'emotions': segment_emotions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
