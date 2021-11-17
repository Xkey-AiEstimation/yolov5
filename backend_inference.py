import time
from typing import List, Dict
import cv2
import torch
import numpy as np
import copy

from models.experimental import attempt_load, Ensemble
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized


def inference(torch_model: Ensemble, cv2_img: np.ndarray, device: torch.device) -> Dict:
    """
    will replace     inference() function in vis_utils.py
    run inference with yolov5 model: torch_model on image cv2_img on the device specified
    Args:
        torch_model:model = attempt_load(weights, map_location=device). dont forget to half() before put it to arg
        cv2_img: image in np array
        device:
    Returns: inference result of the image with the format:
    [{"name": "str", "conf":"float", "bbox":[int, int, int, int]}, ...]

    """
    original_img = copy.deepcopy(cv2_img)
    preferred_img_size = max(cv2_img.shape)

    stride = int(torch_model.stride.max())  # model stride
    img_size_for_inference = check_img_size(preferred_img_size, s=stride)  # check img_size, has to be %32=0

    names = torch_model.module.names if hasattr(torch_model, 'module') else torch_model.names  # name of the symbols
    print(f"model.module.names is: {names}")
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]  # assign colour to each symbol

    cv2_img = letterbox(cv2_img, img_size_for_inference, stride=stride)[0]

    # Convert
    cv2_img = cv2_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    cv2_img = np.ascontiguousarray(cv2_img)
    print(f"shape of cv2_img will be: {cv2_img.shape}")

    # start of the inference
    t0 = time.time()
    torch_img = torch.from_numpy(cv2_img).to(device)
    print(device)
    torch_img = torch_img.half()
    torch_img /= 255.0  # normalize to [0-1]
    if torch_img.ndimension() == 3:
        torch_img = torch_img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = torch_model(torch_img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)  # thresholds subject to change
    t2 = time_synchronized()

    # Process detections
    results = []
    for i, det in enumerate(pred):  # detections per image
        print(f"{i}: {det}")
        if len(det):

            det[:, :4] = scale_coords(torch_img.shape[2:], det[:, :4], original_img.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                print(f"[{xyxy}], confidence: {conf}, class: {cls}")
                bbox = torch.tensor(xyxy).view(1, 4).view(-1).int().tolist()
                symbol_name = names[int(cls)]
                results.append({"name": symbol_name, "conf": float(conf), "bbox": bbox})
    return {"results": results}


def hub_inference(autoshape_model: any, cv2_img: np.ndarray):
    raw_results = autoshape_model(cv2_img)
    results_df = raw_results.pandas().xyxy
    results_df[0].columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'name']
    results_df[0][['xmin', 'ymin', 'xmax', 'ymax']] = results_df[0][['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
    results_df[0]['bbox'] = results_df[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    results = results_df[0][['name', 'conf', 'bbox']].to_dict("records")

    return {'results': results}


if __name__ == "__main__":
    dev = select_device('')
    model = attempt_load("runs/train/exp/weights/last.pt", map_location=dev)  # load FP32 model as ensemble
    model = model.half()
    hub_model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="runs/train/exp/weights/last.pt")

    img = cv2.imread("test_drawings/1.png")
    results = inference(model, img, dev)
    hub_results = hub_inference(hub_model, img)
