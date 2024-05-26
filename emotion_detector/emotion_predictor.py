import librosa  
import numpy as np  
import os
import joblib.externals

# Установка змінної оточення для мінімізації виведення повідомлень TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#=============================================
# Встановлення основної директорії для аудіофайлів
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\audio_files'))

# Шляхи до конкретних аудіофайлів
audio = os.path.join(main_dir, 'Actor_08', '03-01-05-01-02-01-08.wav')
audio1 = os.path.join(main_dir, 'Actor_05', '03-01-06-02-02-02-05.wav')
audio2 = os.path.join(main_dir, 'Actor_23', '03-01-03-01-01-01-23.wav')

# Масив можливих емоцій
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

#============================================
# Функція для екстракції основних характеристик з аудіофайлу
def extract_feature(file_name):
    print('Extracting the features for the file ' + os.path.basename(file_name))  # Вивід інформації про файл
    X, sample_rate = librosa.load(file_name)  # Завантаження файлу
    stft = np.abs(librosa.stft(X))  # Виконання короткочасного Фур'є перетворення
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)  # Обчислення середніх MFCC
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)  # Обчислення середніх значень Chroma
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)  # Обчислення середніх значень Mel
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)  # Обчислення середніх значень Spectral Contrast
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)  # Обчислення середніх значень Tonnetz
    return mfccs, chroma, mel, contrast, tonnetz

#==============================================

# Функція для прогнозування емоції з використанням збереженої моделі
def predict_emotion(model, x):
    clf = joblib.load(model)  # Завантаження моделі
    return(clf.predict(x))  # Повернення результату прогнозу

#==============================================

if __name__ == "__main__":
    
    print("Running " + str(__file__) + "......")  # Вивід повідомлення про запуск скрипта

    # Визначення індексу актуальної емоції з назви файлу
    actual_emo = int(audio2.split('\\')[-1].split('-')[2]) - 1
    features = np.empty((0,193))  # Ініціалізація масиву для характеристик

    mfccs, chroma, mel, contrast, tonnetz = extract_feature(audio2)  # Екстракція характеристик
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])  # Комбінування характеристик в один вектор
    features = np.vstack([features, ext_features])  # Додавання характеристик до масиву

    saved_model = os.path.abspath(os.path.join(os.path.dirname(__file__), './SVM_emotion_clf_24.pkl'))  # Шлях до збереженої моделі
    prediction = predict_emotion(saved_model, features)  # Прогнозування емоції

    print('EMOTIONS :' + str(list(enumerate(emotions))))  # Виведення списку емоцій з їх індексами
    print('Predicted emotion \t Actual emotion ')  # Виведення заголовків для результатів
    print(str(emotions[prediction[0]]) + '\t\t\t ' + str(emotions[actual_emo]))  # Виведення прогнозованої та актуальної емоції
