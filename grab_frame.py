# grab_frame.py
import numpy as np, cv2, pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
align = rs.align(rs.stream.color)

for _ in range(15):
    pipe.wait_for_frames()

frames = align.process(pipe.wait_for_frames())
d, c = frames.get_depth_frame(), frames.get_color_frame()
pipe.stop()
assert d and c, "Нет кадров"

depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
depth_m = np.asanyarray(d.get_data()).astype(np.float32)*depth_scale
color   = np.asanyarray(c.get_data())
depth_clip= np.clip(depth_m, 0, 2.0)
depth_u8  = (255 - (depth_clip/2.0*255)).astype(np.uint8)
depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
cv2.imwrite("color.png", color)
cv2.imwrite("depth_vis.png", depth_vis)
print("Saved color.png, depth_vis.png")