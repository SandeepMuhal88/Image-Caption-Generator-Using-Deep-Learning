# My project
I make a project to generate image captions using deep learning.

## 
### About Project
I make a project to generate image captions using deep learning. and i select pythorch as my framework.because it is have GPU support.and i use ResNet50 as my image encoder and LSTM as my caption decoder.
#
✅ PyTorch works like a charm with:
- Your RTX 4060 (CUDA 11.8/12.1)
- Driver: NVIDIA 550+ (updated)
- Framework: torch, torchvision, torchaudio


## Now starting the building model
### Step one 
### ✅ Setup PyTorch with GPU (Based on Your Card)
Run this in your virtual environment:
    
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### Main Tasks in Phase 2
1. Download & Preprocess Dataset
You already chose Flickr8k – perfect for quick iterations.
```
✅ Download link: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

✅ Captions file: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

```
### Setup the Tensorflow for GPU 
Now i move to the tensorflow because i wamt learn in depth and one think that deployment is very good thats why i am choose the tensorflowSS

### How to setup 
first we need the Tesorflow version that support cuda
### 1. Prerequisites

OS: Windows 10/11 OR Linux (Ubuntu recommended if dual boot/WLS2).

GPU: NVIDIA RTX 4060 (Ada Lovelace, CUDA Compute Capability 8.9).

Python: 3.10 or 3.11 (TensorFlow support is picky).

Conda/venv: Highly recommended (to avoid conflicts).

### 2. Install NVIDIA Drivers

Update your NVIDIA driver to the latest Game Ready / Studio Driver.

Windows: Get from NVIDIA Drivers
.

Linux: Use sudo apt install nvidia-driver-535 (or latest stable).

Verify with:
```
nvidia-smi
```
Should show your RTX XXXX and driver version.

### 3. Install CUDA & cuDNN (Don’t do manually unless you want pain 😅)

TensorFlow now bundles CUDA/cuDNN via pip wheels (no more messy manual installs).
So you just need the right TensorFlow + NVIDIA driver.

TensorFlow 2.16 (latest stable as of 2025) supports CUDA 12.x.

RTX 4060 works fine with CUDA 12.x.

### 4. Create Virtual Environment

Using visual studio (recommended): my case i use windos laptop
```