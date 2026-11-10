import hashlib
import io
import os
import sys
import urllib.request

import numpy as np
import telebot
from keras.models import load_model
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# Веса модели не лежат в git: они выложены в релизе репозитория, чтобы клон оставался лёгким.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/Alexsey111/telegram_colorizer_bot/releases/download/v1.0/model.h5",
)
MODEL_SHA256 = "d8ae091cacd57aaf5a19d56b4c5b506c09d930fbd8a48658dbbe4b223cdd31f5"
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """Возвращает путь к весам, скачивая их при первом запуске.

    Скачивание идёт во временный файл, чтобы оборванная загрузка не оставила
    битый model.h5. После загрузки проверяется sha256.
    """
    if os.path.exists(path):
        return path

    tmp_path = f"{path}.part"
    print(f"Скачиваю веса модели: {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total}% ({done // (1024 * 1024)} МБ)", end="")
    print()

    actual = _sha256(tmp_path)
    if actual != MODEL_SHA256:
        os.remove(tmp_path)
        raise RuntimeError(
            f"Контрольная сумма не совпала: ожидалось {MODEL_SHA256}, получено {actual}. "
            "Файл удалён, попробуйте скачать заново."
        )

    os.replace(tmp_path, path)
    print(f"Веса готовы: {path}")
    return path


TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    sys.exit(
        "Не задан BOT_TOKEN. Запуск: set BOT_TOKEN=<токен от @BotFather> (Windows) "
        "или export BOT_TOKEN=<токен> (Linux/macOS)."
    )

bot = telebot.TeleBot(TOKEN)
model = load_model(ensure_model())


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "Привет! Просто отправьте мне фото, и я попробую его раскрасить.")


def send_result(chat_id, image):
    with io.BytesIO() as output_stream:
        image.save(output_stream, format="PNG")
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)


def generate_colorized_image(grayscale_image, model, original_size):
    img_gray = grayscale_image.resize((256, 256), Image.BILINEAR)
    img_gray_arr = np.array(img_gray, dtype=np.float32)
    img_gray_arr = rgb2lab(np.stack([img_gray_arr, img_gray_arr, img_gray_arr], axis=-1) / 255.0)[:, :, 0]
    img_gray_arr = img_gray_arr.reshape(1, 256, 256, 1)

    # Предсказание модели: два цветовых канала a и b
    output = model.predict(img_gray_arr)
    output = output * 128
    output = np.clip(output, -128, 127)

    result_img = np.zeros((256, 256, 3), dtype=np.float32)
    result_img[:, :, 0] = img_gray_arr[0, :, :, 0]
    result_img[:, :, 1:] = output[0]

    colorized_image = lab2rgb(result_img)
    colorized_image = (colorized_image * 255).astype(np.uint8)
    colorized_image = Image.fromarray(colorized_image)

    # Возвращаем исходный размер кадра
    return colorized_image.resize(original_size, Image.BICUBIC)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_stream = io.BytesIO(downloaded_file)
        original_image = Image.open(image_stream)
        original_size = original_image.size

        grayscale_image = original_image.convert("L")
        colorized_image = generate_colorized_image(grayscale_image, model, original_size)
        send_result(message.chat.id, colorized_image)
    except Exception as exc:  # noqa: BLE001 - бот должен отвечать, а не падать
        bot.reply_to(message, "Произошла ошибка при обработке изображения.")
        print(f"Error: {exc}")


if __name__ == "__main__":
    bot.polling(none_stop=True)
