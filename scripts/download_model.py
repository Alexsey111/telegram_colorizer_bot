"""Скачивание весов модели из релиза репозитория.

Запуск: python scripts/download_model.py [--path model.h5]
Скрипт не требует tensorflow: только стандартная библиотека.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "https://github.com/Alexsey111/telegram_colorizer_bot"
MODEL_URL = f"{REPO}/releases/download/v1.0/model.h5"
MODEL_SHA256 = "d8ae091cacd57aaf5a19d56b4c5b506c09d930fbd8a48658dbbe4b223cdd31f5"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: str) -> str:
    tmp = f"{path}.part"
    print(f"Скачиваю: {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, open(tmp, "wb") as out:
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

    actual = sha256(tmp)
    if actual != MODEL_SHA256:
        os.remove(tmp)
        raise SystemExit(f"Контрольная сумма не совпала: ожидалось {MODEL_SHA256}, получено {actual}")

    os.replace(tmp, path)
    print(f"Готово: {path} ({os.path.getsize(path)} байт)")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать веса модели из релиза")
    parser.add_argument("--path", default="model.h5", help="куда сохранить (по умолчанию model.h5)")
    args = parser.parse_args()

    if os.path.exists(args.path) and sha256(args.path) == MODEL_SHA256:
        print(f"Файл уже на месте и совпадает по сумме: {args.path}")
        return 0

    download(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
