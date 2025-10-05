#!/usr/bin/env python3
import time, sys, traceback
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    print("[FATAL] pyrealsense2 не импортируется. Убедись, что установлен пакет (например, pyrealsense2-macosx) и активирован venv.")
    sys.exit(2)

def try_pipeline(depth=False, color=False, w=640, h=480, fps=30):
    pipe = rs.pipeline()
    cfg  = rs.config()
    if depth:
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    if color:
        # Если BGR8 не взлетит, ниже попробуем YUYV
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    align = None
    if depth and color:
        align = rs.align(rs.stream.color)
    # прогрев
    for _ in range(10):
        frames = pipe.wait_for_frames()
        if align:
            frames = align.process(frames)
    d = frames.get_depth_frame() if depth else None
    c = frames.get_color_frame() if color else None
    pipe.stop()
    return (d is not None) if depth else True, (c is not None) if color else True

def main():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[ERR] Камера не найдена. Проверь USB-кабель/порт. Попробуй другой порт или питаемый хаб.")
        sys.exit(1)

    dev = devs[0]
    name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"
    sn   = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "N/A"
    usb  = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "N/A"
    print(f"[INFO] device: {name}, S/N: {sn}, USB: {usb}")

    # Попробуем аппаратный ресет (часто помогает на macOS)
    try:
        print("[INFO] hardware_reset() …")
        dev.hardware_reset()
        time.sleep(3.0)
    except Exception as e:
        print("[WARN] hardware_reset не выполнен:", e)

    # 1) depth-only
    try:
        ok_d, _ = try_pipeline(depth=True, color=False)
        print(f"[OK] depth-only: {ok_d}")
    except Exception as e:
        print("[ERR] depth-only не запустился:")
        traceback.print_exc()
        ok_d = False

    # 2) color-only (BGR8)
    try:
        _, ok_c = try_pipeline(depth=False, color=True)
        print(f"[OK] color-only (BGR8): {ok_c}")
    except Exception as e:
        print("[WARN] color-only (BGR8) не запустился, пробуем YUYV …")
        # повторим с YUYV
        try:
            pipe = rs.pipeline()
            cfg  = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, 30)
            prof = pipe.start(cfg)
            for _ in range(10): pipe.wait_for_frames()
            pipe.stop()
            ok_c = True
            print("[OK] color-only (YUYV): True")
        except Exception:
            traceback.print_exc()
            ok_c = False

    # 3) оба потока (только если отдельные взлетели)
    both_ok = False
    if ok_d and ok_c:
        try:
            ok_d2, ok_c2 = try_pipeline(depth=True, color=True)
            both_ok = ok_d2 and ok_c2
            print(f"[OK] depth+color: {both_ok}")
        except Exception as e:
            print("[ERR] depth+color не запустились вместе:")
            traceback.print_exc()
    else:
        print("[INFO] Пропускаем depth+color: один из потоков не стартует отдельно.")

    # Резюме рекомендаций
    print("\n===== SUMMARY =====")
    if not ok_d:
        print("- Глубинный поток не стартует: чаще всего USB-питание/кабель/порт. Попробуй другой порт/кабель/питаемый хаб.")
    if not ok_c:
        print("- Цветной поток не стартует: попробуй формат YUYV (мы уже попытались), другой порт/кабель.")
    if ok_d and ok_c and not both_ok:
        print("- Отдельно работают, вместе нет: признак режима USB2. Втыкай в USB3 (5 Gbps) или снизь разрешение/частоту (например 424x240@15).")
    if ok_d and ok_c and both_ok:
        print("- Всё ок: можно запускать основной preview.")
    print("===================\n")

if __name__ == "__main__":
    main()