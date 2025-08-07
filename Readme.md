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