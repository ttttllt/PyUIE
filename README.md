# PyUIE(A Coarse-to-fine Deep Pyramid Network for Underwater Image Enhancement)

This repository contains the PyTorch implementation for  PyUIE, a model designed for underwater image enhancement. If you find this code useful, please consider citing our paper and starring this repository.

## Introduction
Underwater images often suffer from color distortion, reduced contrast, and blurriness due to light refraction, absorption, and scattering. In this paper, we propose a coarse-to-fine deep **Py**ramid network for **U**nderwater **I**mage **E**nhancement (**PyUIE**). Specifically, PyUIE begins by decomposing the input image into high- and low-frequency components using a Laplacian pyramid. The low-frequency residual, which primarily contains lighting and color information, is processed with a lightweight deterministic color mapping network to correct global illumination and color distortions. Concurrently, the high-frequency components containing the fine details are enhanced in a coarse-to-fine manner, such that each higher scale is guided by the reconstruction from the adjacent lower scale. This hierarchical strategy effectively mitigates the risk of over-enhancement by avoiding excessive modifications to the high-frequency components. Additionally, we implement a multi-scale supervised training strategy, enabling the model to learn and reconstruct features across multiple scales, which enhances its ability to capture diverse details and improves its generalization and robustness. Extensive experiments demonstrate that our method successfully restores fine details and small structures in underwater images while producing vivid and visually appealing colors, thereby outperforming existing enhancement methods in both qualitative and quantitative evaluations.

## Installation

To set up the environment and install the required packages, follow these steps:

1. Create a virtual environment:
   ```sh
   conda create -n pyuie python=3.9
   conda activate pyuie
   ```

2. Install PyTorch (v.2.3) and other relevant dependencies:
   ```sh
   conda install pytorch==2.3.0 torchvision==0.17.0 pytorch-cuda=12.2 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```


### Datasets and Model Weights

We provide the model training weights for two datasets，UIEB and LSUI, which can be downloaded via the following Baidu Cloud links:

1. **Dataset 1**: This UIEB dataset is used for training and validation of the model. You can download the corresponding model weights here:

   * [UIEB Model Weights Download Link](https://pan.baidu.com/s/1kngNAaysBdgUsMW9M4j63A?pwd=e6t6)

2. **Dataset 2**: This LSUI dataset is used for training and validation of the model. You can download the corresponding model weights here:

   * [LSUI Model Weights Download Link](https://pan.baidu.com/s/1pjzGLR6yKMRJciYSfS6vLg?pwd=sy4i)

## Download Instructions:

1. Click the respective link.
2. Follow the prompts to complete the file download.
3. Extract the downloaded file and place it in the specified directory in your project.

---


## Testing

1. Download and unzip the code.
2. Place your testing images in the `input_dir` folder.
3. Run `python outpicture.py --initmodel your_model_wright.pt -x input_dir -o output_dir `.


