#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR target + pose (cm) + yaw + smoothing + UDP + Visual Debug (macOS, RealSense D415)
- Фильтр по --name / --id
- Поза и yaw из solvePnP (см) + приведение масштаба к глубине
- Сглаживание по окну, критерий ready
- Публикация JSON по UDP (троттлинг), опция --ready_only
- ВИЗУАЛЬНАЯ ОТЛАДКА: 3D-оси QR, стрелка yaw, хвост, виджет стабильности, статусы
"""

import json, time, sys, re, os, argparse, collections, math, socket
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------- optional deps ----------
try:
    from pyzbar import pyzbar
    HAVE_PYZBAR = True
except Exception:
    HAVE_PYZBAR = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser(description="QR target + smoothing + UDP (cm) with visual debug")
    # target/filter
    p.add_argument("--name", type=str, default=os.getenv("TARGET_NAME","").strip())
    p.add_argument("--id",   type=str, default=os.getenv("TARGET_ID","").strip())
    p.add_argument("--qr_size_cm", type=float, default=float(os.getenv("QR_SIZE_CM", 4.0)))
    # smoothing / readiness
    p.add_argument("--smooth_N", type=int, default=int(os.getenv("SMOOTH_N", 10)))
    p.add_argument("--thr_xy_cm", type=float, default=0.8)
    p.add_argument("--thr_z_cm",  type=float, default=0.5)
    p.add_argument("--thr_yaw_deg", type=float, default=2.0)
    # UDP
    p.add_argument("--udp_host", type=str, default=os.getenv("UDP_HOST", "127.0.0.1"))
    p.add_argument("--udp_port", type=int, default=int(os.getenv("UDP_PORT", "6000")))
    p.add_argument("--ready_only", action="store_true")
    p.add_argument("--pub_rate_hz", type=float, default=float(os.getenv("PUB_RATE_HZ", "20")))
    # viz
    p.add_argument("--viz_off", action="store_true")
    p.add_argument("--axis_cm", type=float, default=3.0, help="axis length to draw (cm)")
    p.add_argument("--trail", type=int, default=20, help="length of center trail")
    return p.parse_args()

# ---------- utils ----------
def colorize_depth(depth_m, max_m=2.0):
    d = np.clip(depth_m, 0.0, max_m)
    d = (d/max_m*255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - d, cv2.COLORMAP_JET)

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

def looks_like_payload(obj: dict) -> bool:
    must = ["id","name","dims_cm","mass_kg","batch","ts"]
    return isinstance(obj, dict) and all(k in obj for k in must)

def sanitize_name(s: str, max_len: int = 80) -> str:
    s = s.replace("\r"," ").replace("\n"," ")
    s = re.sub(r"\s+"," ", s)
    s = "".join(ch for ch in s if ch.isprintable())
    return s[:max_len] if len(s) > max_len else s

RU2LAT = str.maketrans({
    "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Е":"E","Ё":"E","Ж":"Zh","З":"Z","И":"I","Й":"Y",
    "К":"K","Л":"L","М":"M","Н":"N","О":"O","П":"P","Р":"R","С":"S","Т":"T","У":"U","Ф":"F",
    "Х":"Kh","Ц":"Ts","Ч":"Ch","Ш":"Sh","Щ":"Sch","Ы":"Y","Э":"E","Ю":"Yu","Я":"Ya",
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"y",
    "к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f",
    "х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"sch","ы":"y","э":"e","ю":"yu","я":"ya",
})

def draw_text_utf8(bgr, text, org, font_size=20, color=(0,255,0), stroke=2):
    x,y = org
    try:
        if HAVE_PIL:
            from PIL import Image, ImageDraw, ImageFont
            font_paths = [
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
                "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
            font = None
            for p in font_paths:
                if os.path.exists(p):
                    try:
                        font = ImageFont.truetype(p, font_size); break
                    except Exception:
                        pass
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            if stroke>0:
                for dx,dy in [(-stroke,0),(stroke,0),(0,-stroke),(0,stroke)]:
                    draw.text((x+dx,y+dy), text, font=font, fill=(0,0,0))
            draw.text((x,y), text, font=font, fill=(color[2],color[1],color[0]))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    t = text.translate(RU2LAT)
    cv2.putText(bgr, t, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return bgr

def pt_int(x) -> int: return int(np.rint(float(x)))

# ---------- image to BGR ----------
def color_frame_to_bgr_safe(cframe, fmt: str) -> np.ndarray:
    try:
        if fmt=="YUYV":
            w,h = cframe.get_width(), cframe.get_height()
            buf = np.frombuffer(cframe.get_data(), dtype=np.uint8)
            if buf.size == h*w*2:
                yuyv = buf.reshape((h,w,2))
                return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
            elif buf.size == h*w:
                gray = buf.reshape((h,w))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                arr = np.asanyarray(cframe.get_data())
                if arr.ndim==2 and arr.shape==(h,w): return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                if arr.ndim==3 and arr.shape[2]==2: return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
                return arr
        else:
            arr = np.asanyarray(cframe.get_data())
            if arr.ndim==2: return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr
    except Exception:
        return np.empty((0,0,3), dtype=np.uint8)

# ---------- RS modes ----------
MODES = [
    dict(depth=(640,480,30), color=(640,480,30), fmt="BGR8"),
    dict(depth=(640,480,30), color=(640,480,30), fmt="YUYV"),
    dict(depth=(424,240,15), color=(424,240,15), fmt="YUYV"),
]
def start_mode(mode):
    pipe = rs.pipeline(); cfg = rs.config()
    (dw,dh,dfps) = mode["depth"]; (cw,ch,cfps) = mode["color"]
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
    if mode["fmt"]=="BGR8": cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
    else:                   cfg.enable_stream(rs.stream.color, cw, ch, rs.format.yuyv, cfps)
    profile = pipe.start(cfg); return pipe, profile

# ---------- QR decode ----------
_qrd = cv2.QRCodeDetector()
def decode_with_pyzbar(bgr):
    if not HAVE_PYZBAR: return [],[]
    res = pyzbar.decode(bgr); payloads, polys = [],[]
    for r in res:
        x,y,w,h = r.rect
        poly = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
        try: s = r.data.decode("utf-8")
        except Exception:
            try: s = r.data.decode("latin1").encode("latin1").decode("utf-8")
            except Exception: continue
        payloads.append(s); polys.append(poly)
    return payloads, polys

def decode_with_opencv(bgr):
    try:
        ret, decoded, pts, *_ = _qrd.detectAndDecodeMulti(bgr)
        if ret and pts is not None:
            payloads = [s for s in decoded if s]
            polys = [np.array(pts[i]).astype(np.float32) for i,s in enumerate(decoded) if s]
            if payloads: return payloads, polys
    except Exception: pass
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    try:
        ret, decoded, pts, *_ = _qrd.detectAndDecodeMulti(th)
        if ret and pts is not None:
            payloads = [s for s in decoded if s]
            polys = [np.array(pts[i]).astype(np.float32) for i,s in enumerate(decoded) if s]
            if payloads: return payloads, polys
    except Exception: pass
    data, bbox, *_ = _qrd.detectAndDecode(bgr)
    if bbox is not None and data: return [data],[np.array(bbox).astype(np.float32)]
    return [],[]

def robust_decode(bgr):
    if HAVE_PYZBAR:
        p, poly = decode_with_pyzbar(bgr)
        if p: return p, poly, "pyzbar"
    p, poly = decode_with_opencv(bgr)
    if p: return p, poly, "opencv"
    return [], [], "none"

# ---------- filtering ----------
def is_target(obj, want_name: str, want_id: str) -> bool:
    ok_name = True; ok_id = True
    if want_name: ok_name = obj.get("name","").strip().lower() == want_name.strip().lower()
    if want_id:   ok_id   = str(obj.get("id","")).strip() == want_id.strip()
    return ok_name and ok_id

# ---------- geometry ----------
def order_quad_tl_tr_br_bl(quad: np.ndarray) -> np.ndarray:
    pts = quad.reshape(-1,2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def rvec_to_yaw_deg(R: np.ndarray) -> float:
    return float(math.degrees(math.atan2(R[0,2], R[2,2])))

# ---------- smoothing ----------
class PoseSmoother:
    def __init__(self, N=10):
        self.N=N
        self.qx=collections.deque(maxlen=N)
        self.qy=collections.deque(maxlen=N)
        self.qz=collections.deque(maxlen=N)
        self.qyaw=collections.deque(maxlen=N)
        self.qt=collections.deque(maxlen=N)
        self.center_px=collections.deque(maxlen=200)  # для хвоста траектории
    def push(self, x,y,z,yaw, cx=None, cy=None):
        t=time.time(); self.qx.append(x); self.qy.append(y); self.qz.append(z); self.qyaw.append(yaw); self.qt.append(t)
        if cx is not None and cy is not None:
            self.center_px.append((int(cx),int(cy)))
    def stats(self):
        if not self.qx: return None
        arr = lambda q: np.array(q, dtype=np.float32)
        mean = dict(x=float(arr(self.qx).mean()), y=float(arr(self.qy).mean()),
                    z=float(arr(self.qz).mean()), yaw=float(arr(self.qyaw).mean()))
        std  = dict(x=float(arr(self.qx).std(ddof=0)), y=float(arr(self.qy).std(ddof=0)),
                    z=float(arr(self.qz).std(ddof=0)), yaw=float(arr(self.qyaw).std(ddof=0)))
        age_ms = int((time.time()-self.qt[-1])*1000) if self.qt else 0
        return dict(mean=mean, std=std, samples=len(self.qx), age_ms=age_ms)

# ---------- UDP publisher ----------
class UdpPublisher:
    def __init__(self, host: str, port: int, max_hz: float):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.min_dt = 0.0 if max_hz<=0 else 1.0/float(max_hz)
        self.t_last = 0.0
    def try_send(self, payload: dict):
        t = time.time()
        if t - self.t_last < self.min_dt:
            return
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self.sock.sendto(data, self.addr)
            self.t_last = t
        except Exception:
            pass

# ---------- viz helpers ----------
def draw_axes(img, K, dist, rvec, tvec, axis_len_cm=3.0):
    # оси в системе QR (см): X(+) вправо (красный), Y(+) вниз (зелёный), Z(+) от камеры (синий)
    axes_obj = np.float32([[0,0,0],
                           [axis_len_cm,0,0],
                           [0,axis_len_cm,0],
                           [0,0,axis_len_cm]]).reshape(-1,3)
    proj,_ = cv2.projectPoints(axes_obj, rvec, tvec, K, dist)
    p0 = tuple(np.int32(proj[0].ravel()))
    px = tuple(np.int32(proj[1].ravel()))
    py = tuple(np.int32(proj[2].ravel()))
    pz = tuple(np.int32(proj[3].ravel()))
    cv2.arrowedLine(img, p0, px, (0,0,255), 2, tipLength=0.2)   # X red
    cv2.arrowedLine(img, p0, py, (0,255,0), 2, tipLength=0.2)   # Y green
    cv2.arrowedLine(img, p0, pz, (255,0,0), 2, tipLength=0.2)   # Z blue
    return p0

def draw_yaw_arrow(img, cx, cy, yaw_deg, length_px=40, color=(0,255,255)):
    # yaw по камере: 0° вперёд; рисуем проекцию на плоскость кадра (горизонтальная стрелка)
    yaw_rad = math.radians(yaw_deg)
    dx = int(length_px * math.sin(yaw_rad))
    dy = 0
    cv2.arrowedLine(img, (cx,cy), (cx+dx, cy+dy), color, 2, tipLength=0.3)

def draw_stability_widget(img, st, org=(10,65)):
    x0,y0 = org
    txt = f"{'READY' if st['ready'] else 'stabilizing…'}  (std: {st['std']['x']:.2f},{st['std']['y']:.2f},{st['std']['z']:.2f} cm; yaw±{st['std']['yaw']:.1f}°)"
    col = (0,255,0) if st['ready'] else (0,255,255)
    cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    # полоски
    bars = [("x", st["std"]["x"], 1.5), ("y", st["std"]["y"], 1.5), ("z", st["std"]["z"], 1.0), ("yaw", st["std"]["yaw"], 5.0)]
    bx, by = x0, y0+10
    for name, val, scale in bars:
        w = int(min(200, 200*val/scale))
        cv2.rectangle(img, (bx, by), (bx+200, by+10), (60,60,60), 1)
        cv2.rectangle(img, (bx, by), (bx+w,  by+10), col, -1)
        cv2.putText(img, f"{name}:{val:.2f}", (bx+205, by+9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        by += 14

# ---------- main ----------
def main():
    args = parse_args()
    print("[INFO] QR preview (target+smooth+yaw, cm, UDP, viz) starting… q=quit")
    if len(rs.context().query_devices())==0:
        print("[FATAL] Камера не найдена. Проверь USB/кабель/порт."); sys.exit(1)

    smoother = PoseSmoother(args.smooth_N)
    udp = UdpPublisher(args.udp_host, args.udp_port, args.pub_rate_hz)

    cur=0
    lost_counter = 0
    while True:
        mode=MODES[cur]; pipe=profile=None
        try:
            pipe,profile = start_mode(mode)
            align = rs.align(rs.stream.color)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0,      0,       1]], dtype=np.float32)
            distCoeffs = np.zeros((4,1), dtype=np.float32)
            fmt=mode["fmt"]
            print(f"[INFO] camera: {mode['depth']} + {mode['color']} fmt={fmt}  UDP->{args.udp_host}:{args.udp_port}")
        except Exception as e:
            print(f"[WARN] режим {mode} не стартовал: {e}"); cur=(cur+1)%len(MODES); continue

        try:
            miss=0
            while True:
                try:
                    frames = pipe.wait_for_frames(1500)
                    frames = align.process(frames)
                    d,c = frames.get_depth_frame(), frames.get_color_frame()
                    if not d or not c: raise RuntimeError("empty frame")
                except Exception:
                    miss+=1
                    if miss>=5: raise TimeoutError("too many timeouts")
                    continue

                depth_m = np.asanyarray(d.get_data()).astype(np.float32)*depth_scale
                color = color_frame_to_bgr_safe(c, fmt)
                if color.size==0: continue

                banner = f"Filter: name='{args.name}' id='{args.id}'  QR={args.qr_size_cm:.1f} cm  UDP={args.udp_host}:{args.udp_port}"
                cv2.putText(color, banner, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

                payloads, polys, how = robust_decode(color)
                had_target=False
                for poly, payload in zip(polys, payloads):
                    poly = np.array(poly, dtype=np.float32).reshape(-1,2)
                    cx,cy = pt_int(np.mean(poly[:,0])), pt_int(np.mean(poly[:,1]))
                    try:
                        obj = json.loads(payload)
                        if not looks_like_payload(obj): raise ValueError("wrong format")
                    except Exception:
                        poly_i = np.array([[pt_int(x),pt_int(y)] for x,y in poly], dtype=np.int32)
                        cv2.polylines(color, [poly_i], True, (0,140,255), 2)
                        color = draw_text_utf8(color, "BAD/WRONG QR", (cx+6, max(15,cy-6)), color=(0,255,255))
                        continue

                    target = is_target(obj, args.name, args.id) if (args.name or args.id) else True
                    if not target:
                        poly_i = np.array([[pt_int(x),pt_int(y)] for x,y in poly], dtype=np.int32)
                        cv2.polylines(color, [poly_i], True, (0,255,255), 2)
                        continue

                    had_target=True
                    poly_i = np.array([[pt_int(x),pt_int(y)] for x,y in poly], dtype=np.int32)
                    cv2.polylines(color, [poly_i], True, (0,255,0), 2)
                    cv2.circle(color, (cx,cy), 4, (0,255,255), -1)

                    # PnP + depth scale
                    quad_img = order_quad_tl_tr_br_bl(poly)
                    s = args.qr_size_cm
                    obj_pts = np.array([[-s/2, -s/2, 0],
                                        [ s/2, -s/2, 0],
                                        [ s/2,  s/2, 0],
                                        [-s/2,  s/2, 0]], dtype=np.float32)
                    img_pts = quad_img.astype(np.float32).reshape(-1,1,2)
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok:
                        dist_m = d.get_distance(cx, cy)
                        if dist_m<=0: continue
                        X,Y,Z = rs.rs2_deproject_pixel_to_point(intr, [cx,cy], dist_m)
                        x_cm,y_cm,z_cm = X*100.0, Y*100.0, Z*100.0
                        yaw_deg = 0.0
                    else:
                        R,_ = cv2.Rodrigues(rvec)
                        yaw_deg = rvec_to_yaw_deg(R)
                        dist_m = d.get_distance(cx, cy)
                        if dist_m>0:
                            z_depth_cm = dist_m*100.0
                            scale = z_depth_cm / max(1e-6, float(tvec[2,0]))
                            tvec = tvec*scale
                        x_cm,y_cm,z_cm = float(tvec[0,0]), float(tvec[1,0]), float(tvec[2,0])

                        # визуальные оси и стрелка yaw
                        if not args.viz_off:
                            try:
                                p0 = draw_axes(color, K, distCoeffs, rvec, tvec, axis_len_cm=args.axis_cm)
                                draw_yaw_arrow(color, cx, cy, yaw_deg, length_px=40)
                            except Exception:
                                pass

                    name_disp = sanitize_name(str(obj.get("name","?")))
                    color = draw_text_utf8(color, f"{name_disp}  z={z_cm:.1f} cm  yaw={yaw_deg:.1f}°",
                                           (poly_i[0,0], max(15, poly_i[0,1]-8)), color=(0,255,0))

                    # smoothing + publish
                    smoother.push(x_cm,y_cm,z_cm,yaw_deg, cx, cy)
                    st = smoother.stats()
                    if st:
                        sx,sy,sz,syaw = st["std"]["x"], st["std"]["y"], st["std"]["z"], st["std"]["yaw"]
                        ready = (sx<args.thr_xy_cm and sy<args.thr_xy_cm and sz<args.thr_z_cm and syaw<args.thr_yaw_deg)
                        st["ready"]=ready
                        mean = st["mean"]
                        # виджет стабильности
                        if not args.viz_off:
                            draw_stability_widget(color, st, org=(10,60))
                        # хвост
                        if not args.viz_off and len(smoother.center_px)>1:
                            tail = list(smoother.center_px)[-args.trail:]
                            for i in range(1, len(tail)):
                                cv2.line(color, tail[i-1], tail[i], (0,200,255), 2)

                        out = {
                            "stamp": now_iso(),
                            "frame_id": "camera_link",
                            "status": "OK",
                            "id": obj["id"],
                            "name": obj["name"],
                            "dims_cm": obj["dims_cm"],
                            "mass_kg": obj["mass_kg"],
                            "batch": obj["batch"],
                            "ts": obj["ts"],
                            "pose_cm_raw": {"x_cm": float(x_cm), "y_cm": float(y_cm), "z_cm": float(z_cm)},
                            "yaw_deg_raw": float(yaw_deg),
                            "pose_cm_smooth": {"x_cm": mean["x"], "y_cm": mean["y"], "z_cm": mean["z"]},
                            "yaw_deg_smooth": mean["yaw"],
                            "std_cm": {"x": sx, "y": sy, "z": sz},
                            "std_yaw_deg": syaw,
                            "samples": st["samples"],
                            "ready": bool(ready),
                            "age_ms": st["age_ms"],
                            "detector": how,
                            "qr_size_cm": args.qr_size_cm,
                            "pose_from": "PnP+depth_scale"
                        }

                        # файл
                        try:
                            with open("box_target.json", "w", encoding="utf-8") as f:
                                json.dump(out, f, ensure_ascii=False, indent=2)
                            with open("box_name.txt", "w", encoding="utf-8") as f:
                                f.write(str(obj["name"]) + "\n")
                        except Exception:
                            pass

                        # UDP
                        if (not args.ready_only) or ready:
                            udp.try_send(out)

                # LOST / overlay
                if not args.viz_off:
                    if had_target:
                        lost_counter = 0
                    else:
                        lost_counter += 1
                        if lost_counter > 10:
                            cv2.putText(color, "LOST", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3, cv2.LINE_AA)

                # canvas
                depth_vis = colorize_depth(depth_m, 2.0)
                h = min(color.shape[0], depth_vis.shape[0])
                color_r = cv2.resize(color, (int(color.shape[1]*h/color.shape[0]), h))
                depth_r = cv2.resize(depth_vis, (int(depth_vis.shape[1]*h/depth_vis.shape[0]), h))
                canvas = cv2.hconcat([color_r, depth_r])
                cv2.putText(canvas, f"{mode['color']} {mode['fmt']}", (10,25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("QR preview :: color | depth", canvas)
                if (cv2.waitKey(1) & 0xFF)==ord('q'):
                    print("[INFO] quit"); return

        except TimeoutError as e:
            print(f"[WARN] {e}; переключаю режим…"); cur=(cur+1)%len(MODES)
        finally:
            cv2.destroyAllWindows()
            try: pipe.stop()
            except Exception: pass

# ---------- entry ----------
if __name__ == "__main__":
    main()