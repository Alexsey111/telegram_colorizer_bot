import random
import zipfile
import os
import numpy as np
from PIL import Image
import glob
from google.colab import files
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
import shutil
from tqdm import tqdm

uploaded = files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d aayush9753/image-colorization-dataset

# Путь к zip-файлу
zip_file_path = '/content/image-colorization-dataset.zip'

# Папка для распаковки
extracted_folder = '/content'

# Распаковка zip-файла
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Путь к папке с исходными файлами
source_folder = "/content/data/train_color"

# Пути к новым папкам
output_folders = [f"/content/train_color_{i}" for i in range(1, 3)]

# Создание новых папок, если они не существуют
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Получение списка файлов
files = sorted(os.listdir(source_folder))

# Определение размера каждой части
batch_size = len(files) // 2
remainder = len(files) % 2

# Разделение списка файлов на 2 части
file_batches = []
start_idx = 0
for i in range(2):
    end_idx = start_idx + batch_size
    if i < remainder:
        end_idx += 1
    file_batches.append(files[start_idx:end_idx])
    start_idx = end_idx

# Перемещение файлов в соответствующие папки
for i, files_batch in enumerate(file_batches):
    for file in files_batch:
        source_path = os.path.join(source_folder, file)
        output_path = os.path.join(output_folders[i], file)
        shutil.move(source_path, output_path)

print("Разделение файлов завершено.")

def processed_image(img):
    img = img.convert("RGB")  # Убедимся, что изображение в формате RGB
    image = img.resize((256, 256), Image.BILINEAR)
    image = np.array(image, dtype=float)
    size = image.shape
    lab = rgb2lab(1.0/255*image)
    X, Y = lab[:,:,0], lab[:,:,1:]
    Y /= 128    # масштабируем Y к диапазону [-1, 1]
    X = X.reshape(1, size[0], size[1], 1)
    Y = Y.reshape(1, size[0], size[1], 2)
    return X, Y, size

def load_data(path):
    Input, Output = [], []
    images = os.listdir(path)

    for i in tqdm(range(len(images))):
        path_ = os.path.join(path, images[i])
        image = Image.open(path_)
        X, Y, _ = processed_image(image)
        Input.append(X)
        Output.append(Y)

    Input = np.concatenate(Input)
    Output = np.concatenate(Output)

    return Input, Output

dataset_path = "/content/train_color_1"
Input, Output = load_data(dataset_path)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)

# Вывод размерности датасетов
print(f"Input dataset shape: {Input.shape}")
print(f"Output dataset shape: {Output.shape}")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

# Сохранение данных
np.save("/content/X_train.npy", X_train)
np.save("/content/X_test.npy", X_test)
np.save("/content/Y_train.npy", Y_train)
np.save("/content/Y_test.npy", Y_test)

# Загрузка данных
X_train = np.load("/content/X_train.npy")
X_test = np.load("/content/X_test.npy")
Y_train = np.load("/content/Y_train.npy")
Y_test = np.load("/content/Y_test.npy")

def build_adapted_unet(input_shape):
    img_input = Input(input_shape)

    # Сжатие изображения
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    x = MaxPooling2D()(block_1_out)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Уменьшение размера изображения
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Уменьшение размера изображения
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    x = MaxPooling2D()(block_3_out)

    # Уменьшение размера изображения
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)

    x = block_4_out

    # Расширение изображения
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Расширение изображения
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Расширение изображения
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Выходные слои с двумя каналами a и b
    a_channel = Conv2D(1, (3, 3), activation='tanh', padding='same', name='a_channel')(x)
    b_channel = Conv2D(1, (3, 3), activation='tanh', padding='same', name='b_channel')(x)

    output_layer = concatenate([a_channel, b_channel], axis=-1)

    model = Model(img_input, output_layer)

    return model

# Создание модели
input_shape = (256, 256, 1)
model = build_adapted_unet(input_shape)

# Вывод архитектуры модели
model.summary()

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, epochs=75, batch_size=8):
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Определяем путь для сохранения/загрузки
    model_path = 'my_unet_model.h5'

    # Проверяем, существует ли файл модели для загрузки
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded and will continue training...")
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='mse')
        print("Compiled new model...")

    # Настройка callback для сохранения модели
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False  # Сохранение модели
    )

    history = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        callbacks=[lr_reducer, checkpoint_callback])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.show()

    # Прогнозирование на тестовых данных и визуализация
    output = model.predict(X_test)
    output *= 128
    min_vals, max_vals = -128, 127
    ab = np.clip(output[0], min_vals, max_vals)

    cur = np.zeros((X_test.shape[1], X_test.shape[2], 3))
    cur[:,:,0] = np.clip(X_test[0][:,:,0], 0, 100)
    cur[:,:,1:] = ab

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(X_test[0][:,:,0], cmap='gray')
    plt.title('Input Image')

    # Визуализация предсказанного изображения
    plt.subplot(1, 2, 2)
    plt.imshow(lab2rgb(cur[:256, :256, :]))
    plt.title('Predicted Colorized Image')

    plt.show()

# Преобразование меток в необходимый размер
y_train_resized = Y_train[:, :256, :256, :]
y_test_resized = Y_test[:, :256, :256, :]


# активация обучения
train_and_evaluate(model, X_train, y_train_resized, X_test, y_test_resized, epochs=75, batch_size=8)

# Сохранение обученной модели
model.save("my_unet_model.h5")

model = load_model("my_unet_model.h5")

def visualize_predictions(model, black_path='/content/data/test_black', color_path='/content/data/test_color', num_imgs=5):
    black_images = [f for f in os.listdir(black_path) if os.path.isfile(os.path.join(black_path, f))]
    selected_imgs = random.sample(black_images, num_imgs)

    plt.figure(figsize=(15, num_imgs * 3))

    for i, img_name in enumerate(selected_imgs):
        # Загрузка и предобработка черно-белого изображения
        img_gray = Image.open(os.path.join(black_path, img_name))
        img_gray_resized = img_gray.resize((256, 256), Image.BILINEAR)
        img_gray_arr = np.array(img_gray_resized, dtype=float)
        img_gray_arr = rgb2lab(img_gray_arr / 255.0)[:, :, 0]
        img_gray_arr = img_gray_arr.reshape(1, 256, 256, 1)

        # Загрузка соответствующего цветного изображения для сравнения
        img_color = Image.open(os.path.join(color_path, img_name))
        img_color_resized = img_color.resize((256, 256), Image.BILINEAR)

        # Предсказание модели
        output = model.predict(img_gray_arr)
        output *= 128
        output = np.clip(output[0], -128, 127)

        result_img = np.zeros((256, 256, 3))
        result_img[:,:,0] = img_gray_arr[0][:,:,0]
        result_img[:,:,1:] = output

        # Подготовка к выводу
        plt.subplot(num_imgs, 3, i * 3 + 1)
        plt.imshow(img_gray_resized, cmap='gray')
        plt.title('Input')

        plt.subplot(num_imgs, 3, i * 3 + 2)
        plt.imshow(img_color_resized)
        plt.title('Original')

        plt.subplot(num_imgs, 3, i * 3 + 3)
        plt.imshow(lab2rgb(result_img))
        plt.title('Predicted')

    plt.tight_layout()
    plt.show()

# Вызов функции для визуализации
visualize_predictions(model)
