#!/usr/bin/env python3
"""
DeepStream YOLOv8 + Brand/Color/Speed - Clean version
"""
import sys
sys.path.append('../')
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
import numpy as np
import cv2
import time
import json
import os
from datetime import datetime
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import onnxruntime as ort

# --- Config ---
MUXER_OUTPUT_WIDTH = 960
MUXER_OUTPUT_HEIGHT = 540
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720

RTSP_SOURCES = [
    "rtsp://192.168.0.151:8554/stream1",
    "rtsp://192.168.0.151:8554/stream2",
    "rtsp://192.168.0.151:8554/stream1",
    "rtsp://192.168.0.151:8554/stream2",
    "rtsp://192.168.0.151:8554/stream1",
]

CLASSIFY_INTERVAL = 30
MAX_CLASSIFY_PER_FRAME = 4

# --- Brand classifier ---
brand_session = ort.InferenceSession("final_brands.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
BRAND_LABELS = open("labels_brands.txt").read().strip().split("\n")

# --- Color detection ---
COLOR_MAP = [
    ("Black", (0, 0, 0)), ("White", (255, 255, 255)), ("Gray", (128, 128, 128)),
    ("Silver", (192, 192, 192)), ("Red", (220, 20, 60)), ("Maroon", (128, 0, 0)),
    ("Orange", (255, 140, 0)), ("Yellow", (255, 230, 0)), ("Green", (0, 128, 0)),
    ("Dark Green", (0, 100, 0)), ("Light Green", (144, 238, 144)),
    ("Blue", (135, 206, 235)), ("Navy Blue", (0, 0, 128)), ("Gold", (212, 175, 55)),
    ("Pink", (255, 182, 193)), ("Charcoal", (54, 69, 79)), ("Bronze", (140, 120, 83)),
]

# --- Speed ---
LENGTH_MAP = {
    "Motorcycle": 2.0, "Wuling": 3.0, "Mini": 3.8, "Neta": 3.8, "Suzuki": 4.0,
    "Aion": 4.6, "BYD": 4.6, "GWM": 4.6, "Geely": 4.6, "Haval": 4.6,
    "Honda": 4.5, "Jaecoo": 4.5, "MG": 4.5, "Mazda": 4.5,
    "Nissan": 4.6, "Peugeot": 4.5, "Proton": 4.5, "Subaru": 4.6, "Volkswagen": 4.5,
    "BMW": 4.7, "Mercedes-Benz": 4.7, "Tesla": 4.7, "Kia": 4.8, "Toyota": 4.7,
    "Isuzu": 5.2, "Ford": 5.3, "Chevrolet": 5.2, "Mitsubishi": 5.1,
    "Hino": 8.0, "Truck": 9.0, "Bus": 12.0,
}

# --- State ---
track_history = {}
track_cache = {}
track_source = {}
global_frame_count = 0
fps_streams = {}

# --- Logs ---
os.makedirs("logs/frames", exist_ok=True)
os.makedirs("logs/summary", exist_ok=True)
log_files_frames = {}
log_files_summary = {}
FRAME_LOG_INTERVAL = 5


def classify_color(crop_bgr):
    h, w = crop_bgr.shape[:2]
    cy, cx = h // 4, w // 4
    center = crop_bgr[cy:cy + h // 2, cx:cx + w // 2]
    avg_bgr = center.mean(axis=(0, 1))
    avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])
    min_dist = float('inf')
    best_color = "Unknown"
    for name, rgb in COLOR_MAP:
        dist = sum((a - b) ** 2 for a, b in zip(avg_rgb, rgb))
        if dist < min_dist:
            min_dist = dist
            best_color = name
    conf = max(0.0, 1.0 - (min_dist / 50000.0))
    return best_color, conf


def estimate_speed(track_id, cx, cy):
    now = time.time()
    if track_id not in track_history:
        track_history[track_id] = (cx, cy, now)
        return 0.0
    prev_cx, prev_cy, prev_time = track_history[track_id]
    dt = now - prev_time
    if dt < 0.1:
        return 0.0
    dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
    speed = dist / dt
    track_history[track_id] = (cx, cy, now)
    return speed


def calculate_speed_kmh(speed_px_s, bbox_width_px, vehicle_class):
    if bbox_width_px < 10 or speed_px_s < 1:
        return 0.0
    real_length_m = LENGTH_MAP.get(vehicle_class, 4.5)
    meters_per_px = real_length_m / bbox_width_px
    return speed_px_s * meters_per_px * 3.6


def osd_sink_pad_buffer_probe(pad, info, u_data):
    global global_frame_count
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    global_frame_count += 1

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Get frame as numpy
        frame_bgr = None
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_img = np.array(n_frame, copy=True, order='C')
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        except Exception:
            pass

        # Collect crops for classification
        crops_to_classify = []
        obj_list = []

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_list.append(obj_meta)

            if obj_meta.class_id in (2, 3, 5, 7) and frame_bgr is not None:
                tid = obj_meta.object_id
                cached = track_cache.get(tid)
                if not cached or (global_frame_count - cached.get("frame", 0)) >= CLASSIFY_INTERVAL:
                    if len(crops_to_classify) < MAX_CLASSIFY_PER_FRAME:
                        rect = obj_meta.rect_params
                        x1, y1 = max(0, int(rect.left)), max(0, int(rect.top))
                        x2 = min(frame_bgr.shape[1], int(rect.left + rect.width))
                        y2 = min(frame_bgr.shape[0], int(rect.top + rect.height))
                        if x2 > x1 + 32 and y2 > y1 + 32:
                            crops_to_classify.append((tid, frame_bgr[y1:y2, x1:x2], obj_meta.class_id))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Batch classify
        if crops_to_classify:
            car_crops = [(tid, crop) for tid, crop, cls in crops_to_classify if cls == 2]
            brand_results = []
            if car_crops:
                batch = np.stack([cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0 for _, c in car_crops])
                out = brand_session.run(None, {"image": batch})[0]
                for i in range(len(car_crops)):
                    idx = int(np.argmax(out[i]))
                    conf = float(out[i][idx])
                    brand_results.append((BRAND_LABELS[idx], conf) if conf > 0.3 else ("", 0.0))

            car_idx = 0
            for tid, crop, cls in crops_to_classify:
                color_name, color_conf = classify_color(crop)
                brand_name, brand_conf = "", 0.0
                if cls == 2 and car_idx < len(brand_results):
                    brand_name, brand_conf = brand_results[car_idx]
                    car_idx += 1
                track_cache.setdefault(tid, {}).update({
                    "brand": brand_name, "brand_conf": brand_conf,
                    "color": color_name, "color_conf": color_conf,
                    "frame": global_frame_count,
                })

        # Second pass: display + logging
        for obj_meta in obj_list:
            track_id = obj_meta.object_id
            obj_label = obj_meta.obj_label
            brand, color, speed = "", "", 0.0

            if obj_meta.class_id in (2, 3, 5, 7):
                rect = obj_meta.rect_params
                cx = rect.left + rect.width / 2
                cy = rect.top + rect.height / 2
                speed_px = estimate_speed(track_id, cx, cy)

                cached = track_cache.get(track_id)
                if obj_meta.class_id == 2:
                    brand = cached["brand"] if cached and cached.get("brand") else "..."
                    color = cached["color"] if cached and cached.get("color") else "..."
                    vclass = brand if brand != "..." else "car"
                elif obj_meta.class_id == 3:
                    color = cached["color"] if cached and cached.get("color") else "..."
                    vclass = "Motorcycle"
                elif obj_meta.class_id == 5:
                    color = cached["color"] if cached and cached.get("color") else "..."
                    vclass = "Bus"
                elif obj_meta.class_id == 7:
                    color = cached["color"] if cached and cached.get("color") else "..."
                    vclass = "Truck"
                else:
                    vclass = obj_label

                speed = calculate_speed_kmh(speed_px, rect.width, vclass)

                # Track state
                tc = track_cache.setdefault(track_id, {"frame": global_frame_count})
                tc.setdefault("enter_time", time.time())
                tc.setdefault("vehicle_type", obj_label)
                tc.setdefault("speeds", [])
                if speed > 0:
                    tc["speeds"].append(speed)
                tc["last_speed"] = speed

                # Per-frame log
                if global_frame_count % FRAME_LOG_INTERVAL == 0:
                    cam_id = track_source.get(track_id, 0)
                    entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "frame": global_frame_count,
                        "stream_id": cam_id,
                        "track_id": int(track_id),
                        "vehicle_type": obj_label,
                        "brand": tc.get("brand", ""),
                        "color": tc.get("color", ""),
                        "speed_kmh": round(speed, 1),
                        "bbox": [round(rect.left), round(rect.top), round(rect.width), round(rect.height)],
                    }
                    lf = log_files_frames.get(cam_id)
                    if lf is None:
                        name = RTSP_SOURCES[cam_id].rstrip("/").split("/")[-1] if cam_id < len(RTSP_SOURCES) else "cam_%d" % cam_id
                        lf = open("logs/frames/%s_%d.jsonl" % (name, cam_id), "a")
                        log_files_frames[cam_id] = lf
                    lf.write(json.dumps(entry) + "\n")
                    lf.flush()

            # Display text
            parts = [obj_label]
            if obj_meta.class_id == 2 and brand:
                parts.append(brand)
            if color:
                parts.append(color)
            parts.append("ID:%d" % track_id)
            if speed > 2:
                parts.append("%.0fkm/h" % speed)

            # Bbox color
            border = obj_meta.rect_params
            if obj_meta.class_id == 2:
                border.border_color.set(1.0, 0.0, 0.0, 1.0)
            elif obj_meta.class_id == 5:
                border.border_color.set(1.0, 1.0, 0.0, 1.0)
            elif obj_meta.class_id == 7:
                border.border_color.set(0.0, 0.0, 1.0, 1.0)
            elif obj_meta.class_id == 3:
                border.border_color.set(0.0, 1.0, 0.0, 1.0)

            txt = obj_meta.text_params
            txt.display_text = " | ".join(parts)
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 10
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

        # Cleanup stale tracks + summary log
        if global_frame_count % 30 == 0:
            active_ids = {obj.object_id for obj in obj_list}
            for tid in list(track_cache.keys()):
                if tid not in active_ids:
                    tc = track_cache[tid]
                    cam_id = track_source.get(tid, 0)
                    speeds = tc.get("speeds", [])
                    entry = {
                        "timestamp_enter": datetime.fromtimestamp(tc.get("enter_time", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp_exit": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stream_id": cam_id,
                        "track_id": int(tid),
                        "vehicle_type": tc.get("vehicle_type", ""),
                        "brand": tc.get("brand", ""),
                        "brand_conf": round(tc.get("brand_conf", 0.0), 3),
                        "color": tc.get("color", ""),
                        "color_conf": round(tc.get("color_conf", 0.0), 3),
                        "avg_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else 0.0,
                        "max_speed_kmh": round(max(speeds), 1) if speeds else 0.0,
                        "duration_sec": round(time.time() - tc.get("enter_time", time.time()), 1),
                    }
                    lf = log_files_summary.get(cam_id)
                    if lf is None:
                        name = RTSP_SOURCES[cam_id].rstrip("/").split("/")[-1] if cam_id < len(RTSP_SOURCES) else "cam_%d" % cam_id
                        lf = open("logs/summary/%s_%d.jsonl" % (name, cam_id), "a")
                        log_files_summary[cam_id] = lf
                    lf.write(json.dumps(entry) + "\n")
                    lf.flush()
                    del track_cache[tid]
            for tid in list(track_history.keys()):
                if tid not in active_ids:
                    del track_history[tid]
            for tid in list(track_source.keys()):
                if tid not in active_ids:
                    del track_source[tid]

        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def pre_tiler_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                track_source[obj_meta.object_id] = frame_meta.source_id
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gstname = caps.get_structure(0).get_name()
    if gstname.find("video") != -1:
        features = caps.get_features(0)
        if features.contains("memory:NVMM"):
            bin_ghost_pad = data.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if is_aarch64() and name.find("nvv4l2decoder") != -1:
        Object.set_property("bufapi-version", True)
    if name.find("rtspsrc") != -1:
        Object.set_property("latency", 2000)
        Object.set_property("timeout", 5000000)


def create_source_bin(index, uri):
    nbin = Gst.Bin.new("source-bin-%02d" % index)
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin-%02d" % index)
    if not nbin or not uri_decode_bin:
        return None
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
    Gst.Bin.add(nbin, uri_decode_bin)
    nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    return nbin


def main(args):
    num_sources = len(RTSP_SOURCES)
    for i in range(num_sources):
        fps_streams["stream{0}".format(i)] = GETFPS(i)

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)

    for i in range(num_sources):
        print("Creating source bin for %s" % RTSP_SOURCES[i])
        source_bin = create_source_bin(i, RTSP_SOURCES[i])
        if not source_bin:
            return -1
        pipeline.add(source_bin)
        sinkpad = streammux.get_request_pad("sink_%u" % i)
        source_bin.get_static_pad("src").link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    # Properties
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 4000000)

    pgie.set_property("config-file-path", "dstest2cam_pgie_yolov8.txt")
    pgie.set_property("batch-size", num_sources)

    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "iou_config.txt")

    tiler.set_property("rows", 2)
    tiler.set_property("columns", 3)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    nvosd.set_property("process-mode", 0)
    nvosd.set_property("display-text", 1)
    sink.set_property("sync", False)
    sink.set_property("qos", 0)

    # Add + Link
    for elem in [queue1, pgie, queue2, tracker, queue3, tiler, queue4, nvvidconv, queue5, nvosd, queue6, transform, sink]:
        pipeline.add(elem)

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(tiler)
    tiler.link(queue4)
    queue4.link(nvvidconv)
    nvvidconv.link(queue5)
    queue5.link(nvosd)
    nvosd.link(queue6)
    queue6.link(transform)
    transform.link(sink)

    # Event loop
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Probes
    nvvidconv_src_pad = nvvidconv.get_static_pad("src")
    if nvvidconv_src_pad:
        nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if tiler_sink_pad:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, pre_tiler_probe, 0)

    print("Starting pipeline with %d stream(s)..." % num_sources)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
