import telebot
import os
import gdown
import io
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from keras.models import load_model

TOKEN = os.getenv("BOT_TOKEN")
# Создаем экземпляр бота
bot = telebot.TeleBot(TOKEN)

# Загрузка модели, если она не существует локально
model_url = 'https://drive.google.com/uc?id=1--sZIzfnDlm4F7EZtSs7QXhPACOdSO2-'
model_path = 'model.h5'
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
model = load_model(model_path)

# Обработчики сообщений
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Просто отправьте мне фото, и я попытаюсь его раскрасить.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_stream = io.BytesIO(downloaded_file)
        original_image = Image.open(image_stream)
        original_size = original_image.size  # Сохраняем исходный размер изображения

        grayscale_image = original_image.convert('L')
        colorized_image = generate_colorized_image(grayscale_image, model, original_size)
        send_result(message.chat.id, colorized_image)

    except Exception as e:
        bot.reply_to(message, "Произошла ошибка при обработке изображения.")
        print(f"Error: {e}")

def send_result(chat_id, image):
    with io.BytesIO() as output_stream:
        image.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)

def generate_colorized_image(grayscale_image, model, original_size):
    img_gray = grayscale_image.resize((256, 256), Image.BILINEAR)
    img_gray_arr = np.array(img_gray, dtype=np.float32)
    img_gray_arr = rgb2lab(np.stack([img_gray_arr, img_gray_arr, img_gray_arr], axis=-1) / 255.0)[:, :, 0]
    img_gray_arr = img_gray_arr.reshape(1, 256, 256, 1)

    # Предсказание модели
    output = model.predict(img_gray_arr)
    output = output * 128
    output = np.clip(output, -128, 127)

    # Создание полноцветного изображения
    result_img = np.zeros((256, 256, 3), dtype=np.float32)
    result_img[:, :, 0] = img_gray_arr[0, :, :, 0]
    result_img[:, :, 1:] = output[0]

    # Преобразуем L*a*b обратно в RGB
    colorized_image = lab2rgb(result_img)
    colorized_image = (colorized_image * 255).astype(np.uint8)
    colorized_image = Image.fromarray(colorized_image)

    # Изменяем размер обратно к оригинальному
    colorized_image = colorized_image.resize(original_size, Image.BICUBIC)

    return colorized_image

bot.polling(none_stop=True)
