# PyUIE

This repository contains the PyTorch implementation for  PyUIE, a model designed for underwater image enhancement. If you find this code useful, please consider citing our paper and starring this repository.


## Installation

To set up the environment and install the required packages, follow these steps:

1. Create a virtual environment:
   ```sh
   conda create -n cevae python=3.9
   conda activate cevae
   ```

2. Install PyTorch (v.2.3) and other relevant dependencies:
   ```sh
   conda install pytorch==2.3.0 torchvision==0.17.0 pytorch-cuda=12.2 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```


## Testing

1. Download and unzip the code.
2. Place your testing images in the `test_images` folder.
3. Run `python outpicture.py --initmodel your_model_wright.pt -x input_dir -o output_dir `.
