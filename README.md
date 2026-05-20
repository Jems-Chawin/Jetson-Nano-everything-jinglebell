# Jetson-Nano-everything-jinglebell

## Jetson Nano: fill in the YOLO26n model before running

Before starting `deepstream-2cam/deepstream_yolo.py`, set the model paths in the
DeepStream primary inference config.

1. Export or download a YOLO26n ONNX model.
	- Ultralytics docs: https://docs.ultralytics.com/models/yolo26
	- If you use `export_yolov8_ds.py`, change:
	  - `weights = "yolo26n.pt"`
	  - `onnx_output_file = "yolo26n.onnx"`
2. Copy the model files onto the Jetson (example: `/home/<user>/models/yolo26n/`).
3. Edit `deepstream-2cam/dstest2cam_pgie_yolov8.txt` and update:
	- `onnx-file=.../yolo26n.onnx`
	- `model-engine-file=.../yolo26n_b5_gpu0_fp16.engine` (DeepStream will create this)
	- `labelfile-path=.../labels.txt`
	- `custom-lib-path=.../libnvdsinfer_custom_impl_Yolo.so`
4. If an old `.engine` exists for another model, delete it so DeepStream rebuilds.

Then run the app from the `deepstream-2cam` folder.
