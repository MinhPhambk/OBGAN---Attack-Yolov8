import torch
import torchvision
import cv2
import numpy as np
import torch.nn as nn
import cv2
import torch
from yolov8.nn.autobackend import AutoBackend
from yolov8.utils.ops import non_max_suppression

class IdentityLinear(nn.Module):
    def __init__(self, input_size=80):
        super(IdentityLinear, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=True)  # Input size to output size

        # Initialize weight matrix as identity and bias vector as zeros
        self.linear.weight.data = torch.eye(input_size, dtype=torch.float64)
        self.linear.bias.data = torch.zeros(input_size, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)

ROOT = './yolov8'  # YOLOv5 root directory
convert_gradient = IdentityLinear()

def model(dataset=None, conf_thres=0, device=None, one_img=False, label_mode=False):
    if device == None: device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    weights = "yolov8s.pt"
    data = ROOT + '/' + "data/coco128.yaml"
    imgsz=(320, 320)
    resize = torchvision.transforms.Resize(imgsz)

    # Load model
    model = AutoBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    if one_img:
        im = dataset
        # Run inference
        batch = 1
        model.warmup(imgsz=(1 if pt or model.triton else batch, 3, *imgsz))  # warmup

        im = resize(im)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im = im[None]  # expand for batch dimim = im.permute(2, 0, 1) / 255
        

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres)
           
        c_ = 0
        res = np.zeros((batch, 80))
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    confidence = float(conf)
                    if res[c_][c] < confidence:
                        res[c_][c] = confidence

        # print(names[np.argmax(res[c_])])
        return np.argmax(res[c_])


    else:
        if isinstance(dataset, str):        
            # Run inference
            batch = 1
            model.warmup(imgsz=(1 if pt or model.triton else batch, 3, *imgsz))  # warmup

            im = cv2.imread(dataset)
            im = torch.from_numpy(im).to(model.device)
            im = im.permute(2, 0, 1) / 255
            im = resize(im)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im = im[None]  # expand for batch dim
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres=conf_thres)
            c_ = 0
            res = np.zeros((batch, 80))
            for i, det in enumerate(pred):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        confidence = float(conf)
                        if res[c_][c] < confidence:
                            res[c_][c] = confidence

            if label_mode: return names[np.argmax(res[c_])]
            return np.argmax(res[c_])

        else:
            # Run inference
            batch = dataset.shape[0]
            model.warmup(imgsz=(1 if pt or model.triton else batch, 3, *imgsz))  # warmup

            c_ = 0
            res = np.zeros((batch, 80))
            
            for im in dataset:
                im = resize(im)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im = im[None]  # expand for batch dim

                # Inference
                pred = model(im)
                
                # NMS
                pred = non_max_suppression(pred, conf_thres=conf_thres)

                # Process predictions
                for i, det in enumerate(pred):
                    if len(det):
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            confidence = float(conf)
                            if res[c_][c] < confidence:
                                res[c_][c] = confidence

                            # print(c_, label, confidence_str)
                c_ = c_ + 1

            return torch.from_numpy(res).to(device)
    
# def model(dataset=None, conf_thres=0):    
#     device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

#     if not isinstance(dataset, str):   
#         # dzero = torch.zeros_like(dataset)
#         # dzero = nn.Parameter(dzero)
#         # dzero.requires_grad_()
#         # dataset = dataset + dzero
#         return convert_gradient(yolov5_model(dataset, conf_thres=0, device=device).cpu()).to(device)
        
#     return yolov5_model(dataset, conf_thres, device=device)