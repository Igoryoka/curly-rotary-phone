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
    with mp.VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()

# Функція для подавлення шуму
def denoise_audio(y, sr):
    y_denoised = librosa.effects.preemphasis(y)
    return y_denoised

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
        # temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_{uuid.uuid4()}.{file_extension}"
        # file_path = os.path.join(temp_dir, temp_filename)
        file_path = temp_filename
        file.save(file_path)

        # Конвертація відео в аудіо, якщо файл є відео
        if file_extension in ['mp4', 'avi', 'mov']:
            audio_path = f"audio_{uuid.uuid4()}.wav"
            convert_video_to_audio(file_path, audio_path)
            os.remove(file_path)
        else:
            audio_path = file_path

        # Завантаження аудіофайлу
        y, sr = librosa.load(audio_path, sr=None)
        audio_duration = librosa.get_duration(y=y, sr=sr)

        # Опрацювання шуму
        y_denoised = denoise_audio(y, sr)

        # Розділення аудіо на 5 частин
        segment_length = len(y_denoised) // 5

        segments = [y_denoised[i * segment_length:(i + 1) * segment_length] for i in range(5)]


        # Отримання абсолютного шляху до файлу моделі
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'emotion_detector', 'SVM_emotion_clf_24.pkl'))

        # Перевірка наявності файлу моделі
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found at {model_path}'}), 500

        data, samplerate = sf.read(audio_path)
        segment_length = len(data) // 5

        emotions_over_time = []

        # Аналіз кожного сегменту
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            segment = data[i * segment_length:(i + 1) * segment_length]
            sf.write(f'temp_segment_{i}.wav', segment, samplerate)
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(f'temp_segment_{i}.wav')
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            prediction = predict_emotion(model_path, ext_features.reshape(1, -1))
            emotions_over_time.append(emotions[prediction[0]])

            os.remove(f'temp_segment_{i}.wav')

        os.remove(audio_path)

        # Повернення передбачених емоцій
        return jsonify({'emotions': emotions_over_time, 'audio_duration': audio_duration}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
