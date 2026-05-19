#!/usr/bin/env python3
"""
DeepStream RTSP App for Jetson Nano - Display output
"""
import sys
sys.path.append('../')
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720

RTSP_SOURCES = [
    "rtsp://192.168.0.151:8554/stream1",
    "rtsp://192.168.0.151:8554/stream2",
]

fps_streams = {}


def tiler_src_pad_buffer_probe(pad, info, u_data):
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

        obj_counter = {PGIE_CLASS_ID_VEHICLE: 0, PGIE_CLASS_ID_PERSON: 0,
                       PGIE_CLASS_ID_BICYCLE: 0, PGIE_CLASS_ID_ROADSIGN: 0}
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print("Stream=%d Frame=%d Objects=%d Vehicle=%d Person=%d" % (
            frame_meta.pad_index, frame_meta.frame_num,
            frame_meta.num_obj_meta,
            obj_counter[PGIE_CLASS_ID_VEHICLE],
            obj_counter[PGIE_CLASS_ID_PERSON]))

        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

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
            source_bin = data
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if is_aarch64() and name.find("nvv4l2decoder") != -1:
        Object.set_property("bufapi-version", True)
    if name.find("rtspsrc") != -1:
        Object.set_property("latency", 2000)
        Object.set_property("timeout", 5000000)


def create_source_bin(index, uri):
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")
        return None

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin-%02d" % index)
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
        return None

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None

    return nbin


def main(args):
    num_sources = len(RTSP_SOURCES)
    for i in range(num_sources):
        fps_streams["stream{0}".format(i)] = GETFPS(i)

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return -1

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)

    for i in range(num_sources):
        uri = RTSP_SOURCES[i]
        print("Creating source bin for %s" % uri)
        source_bin = create_source_bin(i, uri)
        if not source_bin:
            return -1
        pipeline.add(source_bin)
        sinkpad = streammux.get_request_pad("sink_%u" % i)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not all([queue1, pgie, queue2, sgie, queue3, tiler, queue4, nvvidconv, queue5, nvosd, queue6, transform, sink]):
        sys.stderr.write("Unable to create pipeline elements\n")
        return -1

    # Properties
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 4000000)

    pgie.set_property("config-file-path", "dstest2cam_pgie_config.txt")
    pgie.set_property("batch-size", num_sources)

    sgie.set_property("config-file-path", "dstest2cam_sgie_config.txt")

    tiler.set_property("rows", 1)
    tiler.set_property("columns", 2)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    nvosd.set_property("process-mode", 0)
    nvosd.set_property("display-text", 1)
    sink.set_property("sync", False)
    sink.set_property("qos", 0)

    # Add to pipeline
    for elem in [queue1, pgie, queue2, sgie, queue3, tiler, queue4, nvvidconv, queue5, nvosd, queue6, transform, sink]:
        pipeline.add(elem)

    # Link
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(sgie)
    sgie.link(queue3)
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

    # Probe
    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

    print("Starting pipeline with %d RTSP stream(s)..." % num_sources)
    for i, uri in enumerate(RTSP_SOURCES):
        print("  Stream %d: %s" % (i, uri))

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
