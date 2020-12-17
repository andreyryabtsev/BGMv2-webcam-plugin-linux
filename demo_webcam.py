"""
Videoconferencing plugin demo for Linux.
v4l2loopback-utils needs to be installed, and a virtual webcam needs to be running at `--camera-device` (default: /dev/video1).
A target image and background should be supplied (default: demo_image.png and demo_video.mp4)


Once launched, the script is in background collection mode. Exit the frame and click to collect a background frame.
Upon returning, cycle through different target backgrounds by clicking.
Press Q any time to exit.

Example:
python demo_webcam.py --model-checkpoint "PATH_TO_CHECKPOINT" --resolution 1280 720 --hide-fps
"""

import argparse, os, shutil, time
import numpy as np
import cv2
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from threading import Thread, Lock
from tqdm import tqdm
from PIL import Image
import pyfakewebcam # pip install pyfakewebcam

# --------------- App setup ---------------
app = {
    "mode": "background",
    "bgr": None,
    "bgr_blur": None,
    "compose_mode": "plain",
    "target_background_frame": 0
}

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Virtual webcam demo')

parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
parser.add_argument('--target-video', type=str, default='./demo_video.mp4')
parser.add_argument('--target-image', type=str, default='./demo_image.jpg')
parser.add_argument('--camera-device', type=str, default='/dev/video1')
args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'));
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio
    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()
    def get(self):
        return self._avg_fps

# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        self.webcam = None
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        if self.webcam is not None:
            image_web = np.ascontiguousarray(image, dtype=np.uint8) # .copy()
            image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
            self.webcam.schedule_frame(image_web)
        # else:
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


class Controller: # A cv2 window with a couple buttons for background capture and cycling through target background options
    def __init__(self):
        self.name = "RTHRBM Control"
        self.controls = [
            {
                "type": "button",
                "name": "mode_switch",
                "label": "Grab background",
                "x": 50,
                "y": 20,
                "w": 300,
                "h": 40
            },
            {
                "type": "button",
                "name": "compose_switch",
                "label": "Compose: plain white",
                "x": 50,
                "y": 100,
                "w": 300,
                "h": 40
            }
        ]

        cv2.namedWindow(self.name)
        cv2.moveWindow(self.name, 200, 200)
        cv2.setMouseCallback(self.name, self._raw_process_click)
        self.render()

    def render(self):
        control_image = np.zeros((160,400), np.uint8)
        for button in self.controls:
            control_image[button["y"]:button["y"] + button["h"],button["x"]:button["x"] + button["w"]] = 180
            cv2.putText(control_image, button["label"], (button["x"] + 10, button["y"] + button["h"] // 2 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.imshow(self.name, control_image)
        
    def clicked(self, control):
        if control["name"] == "mode_switch":
            if app["mode"] == "background":
                grab_bgr()
                app["mode"] = "stream"
                control["label"] = "Select another background"
            else:
                app["mode"] = "background"
                control["label"] = "Grab background"
        elif control["name"] == "compose_switch":
            cycle = [("plain", "Compose: plain white"), ("gaussian", "Compose: blur background"), ("video", "Compose: Winter holidays"), ("image", "Compose: Mt. Rainier")]
            current_idx = next(i for i, v in enumerate(cycle) if v[0] == app["compose_mode"])
            next_idx = (current_idx + 1) % len(cycle)
            app["compose_mode"] = cycle[next_idx][0]
            control["label"] = cycle[next_idx][1]
        self.render()
    

    def _raw_process_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for control in self.controls:
                if x > control["x"] and x < control["x"] + control["w"] and y > control["y"] and y < control["y"] + control["h"]:
                    self.clicked(control)
                    
class VideoDataset(Dataset):
    def __init__(self, path: str, transforms: any = None):
        self.cap = cv2.VideoCapture(path)
        self.transforms = transforms
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.cap.read()
        if not ret:
            raise IndexError(f'Idx: {idx} out of length: {len(self)}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()


# --------------- Main ---------------


# Load model
model = torch.jit.load(args.model_checkpoint)
model.backbone_scale = args.model_backbone_scale
model.refine_mode = args.model_refine_mode
model.refine_sample_pixels = args.model_refine_sample_pixels
model.model_refine_threshold = args.model_refine_threshold
model.cuda().eval()


width, height = args.resolution
cam = Camera(width=width, height=height)
dsp = Displayer('RTHRBM Preview', cam.width, cam.height, show_info=(not args.hide_fps))
ctr = Controller()
fake_camera = pyfakewebcam.FakeWebcam(args.camera_device, cam.width, cam.height)
dsp.webcam = fake_camera

def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()

preloaded_image = cv2_frame_to_cuda(cv2.imread(args.target_image))
tb_video = VideoDataset(args.target_video, transforms=ToTensor())

def grab_bgr():
    bgr_frame = cam.read()
    bgr_blur = cv2.GaussianBlur(bgr_frame.astype('float32'), (67, 67), 0).astype('uint8') # cv2.blur(bgr_frame, (10, 10))
    app["bgr"] = cv2_frame_to_cuda(bgr_frame)
    app["bgr_blur"] = cv2_frame_to_cuda(bgr_blur)


def app_step():
    if app["mode"] == "background":
        frame = cam.read()
        key = dsp.step(frame)
        if key == ord('q'):
            return True
    else:
        frame = cam.read()
        src = cv2_frame_to_cuda(frame)
        pha, fgr = model(src, app["bgr"])[:2]
        if app["compose_mode"] == "plain":
            tgt_bgr = torch.ones_like(fgr)
        elif app["compose_mode"] == "image":
            tgt_bgr = nn.functional.interpolate(preloaded_image, (fgr.shape[2:]))
        elif app["compose_mode"] == "video":
            vidframe = tb_video[app["target_background_frame"]].unsqueeze_(0).cuda()
            tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
            app["target_background_frame"] += 1
        elif app["compose_mode"] == "gaussian":
            tgt_bgr = app["bgr_blur"]
            
        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            
        key = dsp.step(res)

        if key == ord('q'):
            return True

with torch.no_grad():
    while True:
        if app_step():
            break
