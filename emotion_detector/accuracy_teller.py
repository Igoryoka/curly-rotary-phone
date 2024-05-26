# Імпортуємо необхідні бібліотеки для роботи з файлами, виконання математичних операцій, функцій машинного навчання та серіалізації моделей.
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib.externals

# Визначаємо константи для поточної директорії, назви моделі, файлу особливостей та файлу міток.
CUR_DIR = os.getcwd()
model_name = 'SVM_emotion_clf_24.pkl'
features = 'Features_emotions_24.npy'
labels = 'labels_emotions_24.npy'

# Функція для завантаження попередньо навченої моделі з файлової системи.
def load_model():
    return joblib.load(os.path.join(CUR_DIR, model_name))

# Функція для виконання прогнозу за допомогою класифікатора та виведення рівня точності.
def fPrediction(clf, train_x, test_x, train_y, test_y):
    prediction = clf.predict(train_x)
    print("Рівень точності детектора емоцій: " + str(accuracy_score(train_y, prediction)))

# Функція для виведення розмірів масивів numpy, корисна для налагодження.
def print_shape(*args):
    for each in args:
        print("Форма: " + str(each.shape))

# Основний блок виконання, працює тільки коли скрипт виконується безпосередньо (не імпортований).
if __name__ == '__main__':
    x = np.load(features)  # Завантаження даних особливостей з файлу .npy
    y = np.load(labels)    # Завантаження даних міток з файлу .npy
    
    # Розділення даних на навчальні та тестові набори, з 90% даних у тестовому наборі.
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.9, random_state=42)

    # Конвертація міток з one-hot кодування в цілі числа класів.
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)

    # Завантаження попередньо навченої моделі
    clf = load_model()
    
    # Виклик функції для виконання прогнозу та виведення рівня точності
    fPrediction(clf, train_x, test_x, train_y, test_y)
