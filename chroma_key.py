from colorsys import rgb_to_hsv

import cv2
import numpy as np
from PIL import Image

HSVRange = tuple[tuple[int, int, int], tuple[int, int, int]]


def chroma_key_opencv(foreground_path: str, background_path: str,
                      output_path: str, hsv_range: HSVRange) -> None:
    foreground = cv2.imread(foreground_path)
    background = cv2.imread(background_path)

    if foreground is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {foreground_path}")
    if background is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {background_path}")

    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_range[0], dtype=np.uint8)
    upper = np.array(hsv_range[1], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    result = foreground.copy()
    result[mask > 0] = background[mask > 0]
    cv2.imwrite(output_path, result)


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return int(round(h * 179)), int(round(s * 255)), int(round(v * 255))


def chroma_key_native(foreground_path: str, background_path: str,
                      output_path: str, hsv_range: HSVRange) -> None:
    foreground = Image.open(foreground_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")
    background = background.resize(foreground.size)

    fg_pixels = foreground.load()
    bg_pixels = background.load()
    result = Image.new("RGB", foreground.size)
    result_pixels = result.load()

    (h_min, s_min, v_min), (h_max, s_max, v_max) = hsv_range
    width, height = foreground.size

    for y in range(height):
        for x in range(width):
            r, g, b = fg_pixels[x, y]
            h, s, v = _rgb_to_hsv(r, g, b)

            if (h_min <= h <= h_max and
                    s_min <= s <= s_max and
                    v_min <= v <= v_max):
                result_pixels[x, y] = bg_pixels[x, y]
            else:
                result_pixels[x, y] = (r, g, b)

    result.save(output_path)
