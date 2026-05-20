import os
import sys
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolov8_export(weights, device, fuse=True):
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    if fuse:
        model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def main():
    import warnings
    warnings.filterwarnings("ignore")

    weights = "yolo26n.pt"
    print(f"Starting: {weights}")

    device = torch.device("cpu")
    model = yolov8_export(weights, device)

    model = nn.Sequential(model, DeepStreamOutput())

    onnx_input_im = torch.zeros(1, 3, 320, 320).to(device)
    onnx_output_file = "yolov8n.onnx"

    print("Exporting to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )

    print("Simplifying")
    import onnxslim
    model_onnx = onnx.load(onnx_output_file)
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}")


if __name__ == "__main__":
    main()
