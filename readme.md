The goal of this repository is to supplement the [main Real-Time High Resolution Background Matting repo](https://github.com/PeterL1n/BackgroundMattingV2) with a working demo of a videoconferencing plugin (e.g. the code used in our promotional demos).

# Prerequisites
This plugin requires Linux, because it relies on the [v4l2loopback kernel module](https://github.com/umlaeute/v4l2loopback) to create and stream to virtual video devices. We welcome and encourage community adaptations to other platforms.

1. Install v4l2loopback. On Debian/Ubuntu, the command is likely `sudo apt-get install v4l2loopback-utils`.
2. Install required packages in `requirements.txt`
   - If starting from the main repository, just install [pyfakewebcam](https://github.com/jremmons/pyfakewebcam) (Python interface for v4l2loopback, `pip install pyfakewebcam`). You can move the script to the root level in that repository.

# Directions
Before running the plugin, the virtual web camera device needs to be created. 
```
sudo modprobe v4l2loopback devices=1
```
The above command should create a single virtual webcam at `/dev/video1` (the number may change), which is the default stream output for the plugin script. This webcam can now be selected by software such as Zoom, browsers, etc.

After downloading the TorchScript weights of your choice [here](https://drive.google.com/drive/u/1/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh), launch the pluging with e.g.:
```python demo_webcam.py --model-checkpoint torchscript_resnet50_fp32.pth```

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

# Community Contributions
1. [@kevinlester](https://github.com/kevinlester/BGMv2-webcam-plugin-linux) added many realtime settings to the UI controls
2. [@KostiaChorny](https://github.com/KostiaChorny/BGMv2-webcam-plugin-linux) added Windows OBS virtual camera support (not merged but please visit their fork)

Thanks to [CK FreeVideoTemplates](https://www.youtube.com/watch?v=DHRUNWdf3ms) on YouTube for the seasonal video target background.