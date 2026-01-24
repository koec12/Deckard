# Cylinder Tracking System

A Python application for detecting and tracking colored cylinders on a conveyor belt using YOLO object detection and Modbus TCP communication for interfacing with a PLC.

This project is under development and was created as part of the author's studies main project. Use at your own risk; APIs and code paths may change while developing. If you're not referred here by Koen directly, this is probably not what you are looking for.

## Features

- **ROI Selection**: Select up to 9 regions of interest (ROIs) for monitoring
- **Object Detection**: Uses Ultralytics YOLO segmentation model for detection
- **Custom Tracking**: Robust object tracking with class locking and speed calculation
- **Modbus TCP Integration**: Exposes detection status via Modbus TCP holding registers
- **Real-time Visualization**: Live preview with color-coded bounding boxes

## Setup

### 1. Create Virtual Environment

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Settings

Edit `config.yaml` to adjust camera settings, model path, ROI colors, tracker parameters, and Modbus server settings.

### 4. Download YOLO Model

The application expects a custom YOLO segmentation model. You need to download it manually and specify a custom model path in `config.yaml`.

## Usage

### Running the Application

```bash
python app.py
```

### ROI Selection

1. When the application starts, you'll see the camera feed
2. Press `s` to start selecting ROIs
3. Click and drag to select a region of interest
4. Press `Enter` to confirm the ROI, or `Esc` to cancel
5. Repeat for additional ROIs (up to 9)
6. Press `q` to finish ROI selection and start tracking

### Controls

- `s`: Start/continue ROI selection
- `q`: Finish ROI selection and start tracking (or quit if already tracking)
- `c`: Reset tracker
- `Esc`: Cancel current ROI selection
- `Enter`: Confirm current ROI selection

### Modbus TCP Registers

#### ROI Status Registers (11-19)
- Register 11: ROI 1 status
- Register 12: ROI 2 status
- ...
- Register 19: ROI 9 status

Bit mapping:
- Bit 0: Red cylinder detected
- Bit 1: Grey/Silver cylinder detected
- Bit 2: Black cylinder detected
- Bits 3-15: Reserved

#### Object Information Registers
Each detected object in a ROI gets 10 holding registers:
- ROI 1, Object 0: Registers 1000-1009
- ROI 1, Object 1: Registers 1010-1019
- ROI 2, Object 0: Registers 2000-2009
- etc.

Register layout (xyy0-xyy9 where x=ROI, yy=object index):
- xyy0: Object ID
- xyy1: Class ID (0=Black, 1=Red, 2=Silver)
- xyy2: Confidence (0-100%)
- xyy3: Speed (cm/s)
- xyy4: X-coordinates centroid
- xyy5: Y-coordinates centroid
- xyy6-xyy9: Reserved

## Configuration

Edit `config.yaml` to customize:
- Camera source and resolution
- YOLO model path and thresholds
- ROI colors
- Tracker parameters (class locking thresholds, speed calculation)
- Modbus server host and port
- Image preprocessing settings

## Requirements

- Python 3.8+
- Camera or video source
- YOLO segmentation model trained on cylinders

## Notes

- The tracker locks the class of detected objects once confidence exceeds the configured threshold
- Class changes are only allowed if the new class confidence exceeds a higher threshold for multiple consecutive frames
- Objects are reidentified if briefly lost using distance-based matching
- Speed is calculated based on pixel movement and the configured pixels-per-cm ratio

This code is a work in progress and is heavily vibe-coded. Expect API changes, refactors, or added features.
Created for the author's studies main project; not production-ready.

## Optimizing for CUDA (INT8 Quantization)

INT8 quantization can boost inference speed with a small accuracy trade-off. Perform this on the final target hardware.

1. Update GPU drivers.
2. Check CUDA version: `nvidia-smi`.
3. Install the matching PyTorch build from [pytorch.org](https://pytorch.org/).
   - Example: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
4. Install TensorRT: `pip install tensorrt`.
5. Prepare a dataset YAML that points to `valid` set with images + labels (INT8 needs lots of data).
   - Tip: Move train/test images + labels into `valid` to increase samples.
6. Export to TensorRT:

```python
from ultralytics import YOLO

model = YOLO(r"C:\Path\to\model.pt")

model.export(
    format="engine",
    dynamic=True,
    batch=1,
    workspace=None,
    int8=True,
    data=r"C:\Path\to\data.yaml"
)
```

The output `.engine` file replaces the `.pt` model. Update `config.yaml` accordingly.

## License

No license provided. Use for study purposes only.