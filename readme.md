The goal of this repository is to supplement the [main Real-Time High Resolution Background Matting repo](https://github.com/PeterL1n/BackgroundMattingV2) with a working demo of a videoconferencing plugin (e.g. the code used in our promotional demos).

# Prerequisites

## Linux

This plugin relies on the [v4l2loopback kernel module](https://github.com/umlaeute/v4l2loopback) to create and stream to virtual video devices. We welcome and encourage community adaptations to other platforms.

1. Install v4l2loopback. On Debian/Ubuntu, the command is likely `sudo apt-get install v4l2loopback-utils`.
2. Install required packages in `requirements.txt`
   - If starting from the main repository, just install [pyfakewebcam](https://github.com/jremmons/pyfakewebcam) (Python interface for v4l2loopback, `pip install pyfakewebcam`). You can move the script to the root level in that repository.
   
## Windows

For Windows, you can use [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) module for output video stream to OBS Virtual Camera.

1. Install required packages in `requirements.txt` (pyfakewebcam can be skipped in this case).
2. Install `pyvirtualcam`
   - Clone repository `git clone https://github.com/letmaik/pyvirtualcam.git`
   - Build module `python setup.py build`. For this step [Microsoft Visual C++ 14.0 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) are needed
   - Install module `python setup.py install`
3. Install [OBS Studio](https://obsproject.com/ru/download) and [Virtual Camera Plugin](https://github.com/Fenrirthviti/obs-virtual-cam/releases) 

# Directions

## Linux

Before running the plugin, the virtual web camera device needs to be created. 
```
sudo modprobe v4l2loopback devices=1
```
The above command should create a single virtual webcam at `/dev/video1` (the number may change), which is the default stream output for the plugin script. This webcam can now be selected by software such as Zoom, browsers, etc.

## Windows

Ensure, that OBS is installed correctly and virtual camera is detected by system. You can use Skype or any other application to check it out.

## Launch

After downloading the TorchScript weights of your choice [here](https://drive.google.com/drive/u/1/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh), launch the pluging with e.g.:

Linux:
```python demo_webcam.py --model-checkpoint torchscript_resnet50_fp32.pth```


Windows:
```python demo_webcam_win.py --model-checkpoint torchscript_resnet50_fp32.pth```


Once the plugin is launched, a simple OpenCV-based UI will show two buttons:
- Toggle between background selection view and (after snapping a background) actual matting
- Cycle through target background options:
  1. Plain white
  2. Gaussian blur of the background frame
  3. Supplied target video background (samples included in this repo)
  4. Supplied target image background
At any point, press Q to exit.

# Guidelines
For best results:
1. Avoid blocking bright light, especially during background capture.
2. Avoid changing light conditions. For prolonged use, it may be helpful to re-take the background as light conditions drift.

Thanks to [CK FreeVideoTemplates](https://www.youtube.com/watch?v=DHRUNWdf3ms) on YouTube for the seasonal video target background.
