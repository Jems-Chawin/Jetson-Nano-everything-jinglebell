# DeepStream 2-Camera USB App (Jetson Nano)

DeepStream Python app that runs object detection (vehicle, person, bicycle, road sign) on 2 USB cameras simultaneously with a tiled display output.

## Pipeline

```
v4l2src (cam0) ─┐
                ├─► nvstreammux ─► nvinfer ─► nvtiler ─► nvvideoconvert ─► nvdsosd ─► display
v4l2src (cam1) ─┘
```

## Prerequisites

- Jetson Nano with JetPack 4.4+ (DeepStream 5.0+)
- DeepStream SDK installed at `/opt/nvidia/deepstream/deepstream/`
- DeepStream Python bindings (`pyds`) installed
- 2 USB cameras (e.g., Logitech webcams)

### Install DeepStream Python bindings

```bash
# Download from NVIDIA (match your DeepStream version)
# For DS 5.0 on Jetson Nano:
cd /opt/nvidia/deepstream/deepstream/lib
pip3 install pyds-1.0.0-py3-none-linux_aarch64.whl
```

## Usage

```bash
cd deepstream-2cam
python3 deepstream_2cam.py /dev/video0 /dev/video1
```

### Check available cameras

```bash
ls /dev/video*
# or
v4l2-ctl --list-devices
```

## File Structure

```
deepstream-2cam/
├── common/
│   ├── __init__.py
│   ├── FPS.py
│   ├── bus_call.py
│   └── is_aarch_64.py
├── deepstream_2cam.py
├── dstest2cam_pgie_config.txt
└── README.md
```

## Notes

- First run will take a few minutes to generate the TensorRT engine file
- Uses INT8 inference mode for best performance on Jetson Nano
- The tiled display shows both cameras side-by-side (1x2 grid)
- Detection classes: Vehicle, Bicycle, Person, Road Sign
- FPS is printed to console every 5 seconds per stream

## Troubleshooting

- **Camera not found**: Check device paths with `v4l2-ctl --list-devices`
- **Low FPS**: Reduce resolution by editing `MUXER_OUTPUT_WIDTH/HEIGHT` in the script
- **Engine file error**: Delete any existing `.engine` file and let it regenerate
