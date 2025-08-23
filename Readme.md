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

# 1) go to your project directory (or create it)
mkdir -p ~/projects/image-captioner && cd ~/projects/image-captioner

# 2) create a virtual env in the project (named .venv)
python3.11 -m venv .venv        # or: python3.10 -m venv .venv
source .venv/bin/activate

# 3) upgrade pip
pip install --upgrade pip

# 4) install TensorFlow with GPU (pulls compatible CUDA/cuDNN automatically)
pip install "tensorflow[and-cuda]"

# 5) verify GPU is visible
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# 6) (optional) freeze deps
pip freeze > requirements.txt
