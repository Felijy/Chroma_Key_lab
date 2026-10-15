from argparse import ArgumentTypeError, ArgumentParser
import time
from typing import Tuple

from chroma_key import chroma_key_native, chroma_key_opencv

HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


def parse_hsv_range(value: str) -> HSVRange:
    try:
        lower, upper = value.split(";")
        lower_values = tuple(map(int, lower.split(",")))
        upper_values = tuple(map(int, upper.split(",")))
    except ValueError as exc:
        raise ArgumentTypeError(
            "Формат интервала: Hmin,Smin,Vmin;Hmax,Smax,Vmax"
        )

    if len(lower_values) != 3 or len(upper_values) != 3:
        raise ArgumentTypeError("Нужно указать 3 значения в каждой границе интервала")

    if not all(0 <= x <= 255 for x in lower_values + upper_values):
        raise ArgumentTypeError("Все значения должны быть от 0 до 255")
    if not 0 <= lower_values[0] <= 179 or not 0 <= upper_values[0] <= 179:
        raise ArgumentTypeError("H должен быть от 0 до 179 в шкале OpenCV")

    return lower_values, upper_values


def run_images(foreground: str, background: str, hsv_range: HSVRange) -> None:
    opencv_path = "result_opencv.png"
    native_path = "result_native.png"

    start = time.perf_counter()
    chroma_key_opencv(foreground, background, str(opencv_path), hsv_range)
    opencv_time = time.perf_counter() - start

    start = time.perf_counter()
    chroma_key_native(foreground, background, str(native_path), hsv_range)
    native_time = time.perf_counter() - start

    print(f"OpenCV: {opencv_time:.6f} c")
    print(f"Native Python: {native_time:.6f} c")
    print(f"Разница: {native_time / opencv_time:.2f}x")
    print(f"Результаты сохранены")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Простая реализация chroma key")

    parser.add_argument("--foreground", required=True, help="Изображение с цветным фоном")
    parser.add_argument("--background", required=True, help="Второе изображение")
    parser.add_argument("--hsv", required=True, type=parse_hsv_range,
                            help="Интервал HSV: Hmin,Smin,Vmin-Hmax,Smax,Vmax")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_images(args.foreground, args.background, args.hsv)

if __name__ == "__main__":
    main()
