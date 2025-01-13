# Audiogmentation

![Python](https://img.shields.io/badge/python-3.10.16-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-ee4c2c.svg?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.26.4-013243.svg?logo=numpy&logoColor=white)
![Librosa](https://img.shields.io/badge/librosa-0.10.2-yellow.svg?logo=python&logoColor=white)

A sophisticated music genre classification system enhanced with audio data augmentation techniques. This project comes is a newer  and productive-oriented version of my undergraduate thesis. It leverages multiple audio processing libraries and machine learning frameworks to perform accurate genre classification while employing audio augmentation methods to improve model robustness and performance.

## Features

- Music genre classification using deep learning
- Advanced audio data augmentation techniques
- Built with modern Python package management using `uv`
- Comprehensive audio processing capabilities
- Support for various audio formats and transformations

## Dataset

This project uses the dataset introduced in the paper "Automatic Musical Genre Classification Of Audio Signals". If you use this dataset in your research, please cite:

```bibtex
@misc{tzanetakis_essl_cook_2001,
    author    = "Tzanetakis, George and Essl, Georg and Cook, Perry",
    title     = "Automatic Musical Genre Classification Of Audio Signals",
    url       = "http://ismir2001.ismir.net/pdf/tzanetakis.pdf",
    publisher = "The International Society for Music Information Retrieval",
    year      = "2001"
}
```

## Tech Stack

- **Python**: Core programming language
- **PyTorch & Torchaudio**: Deep learning framework and audio processing
- **Librosa**: Audio and music processing
- **Audiomentations**: Audio data augmentation
- **Scikit-learn**: Machine learning utilities
- **Numpy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization

## Prerequisites

- Python 3.8 or higher
- `uv` package manager
- CUDA-compatible GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nachojeda/audiogmentation.git
cd audiogmentation
```

2. Create a new virtual environment using `uv`:
```bash
uv venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

4. Install dependencies using `uv`:
```bash
uv pip install -r requirements.txt
```

##