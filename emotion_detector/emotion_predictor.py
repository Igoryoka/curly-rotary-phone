import librosa  
import numpy as np  
import os
import joblib.externals

# Приховування повідомлень TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Основна директорія з аудіофайлами
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\Audio_Speech_Actors_01-24'))

# Шляхи до аудіофайлів для тестування
audio = os.path.join(main_dir, 'Actor_08', '03-01-05-01-02-01-08.wav')
audio1 = os.path.join(main_dir, 'Actor_05', '03-01-06-02-02-02-05.wav')
audio2 = os.path.join(main_dir, 'Actor_23', '03-01-03-01-01-01-23.wav')

# Шлях до збереженої моделі
saved_model = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SVM_emotion_clf_24.pkl'))

# Список емоцій
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Функція для вилучення ознак з аудіофайлу
def extract_feature(file_name):
    print('Extracting the features for the file ' + os.path.basename(file_name))
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

# Функція для передбачення емоції за допомогою моделі
def predict_emotion(model_path, features):
    model = joblib.load(model_path)
    return model.predict(features)

# Головна частина скрипта
if __name__ == "__main__":
    print("Running "+ str(__file__) + "......")

    # Визначення фактичної емоції з назви файлу
    actual_emo = int(audio2.split('\\')[-1].split('-')[2])-1
    features = np.empty((0,193))

    # Вилучення ознак з аудіофайлу
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(audio2)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])

    # Передбачення емоції за допомогою завантаженої моделі
    prediction = predict_emotion(saved_model, features)

    # Виведення результатів
    print('EMOTIONS :' + str(list(enumerate(emotions))))
    print('Predicted emotion \t Actual emotion ')
    print(str(emotions[prediction[0]]) + '\t\t\t ' + str(emotions[actual_emo]))
