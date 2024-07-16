# ZS-Pan v1
Pan v2 (updated on 5/29/2024)
## Updates
- Introduced the code 'run.py' for seamless one-click execution of all codes
- Updated the codes main_rsp.py, main_sde.py, main_fug.py, and test.py to facilitate unified parameter tuning within 'run.py', streamlining the tuning process
- Adjusted certain default hyperparameters to enhance runtime performance while maintaining consistent metrics (reducing the total time to approximately one minute)

# Get Strarted
## Dataset
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). The downloaded data can be placed everywhere because we do not use relative path. Besides, we recommend the h5py format, as if using the mat format, the data loading section needs to be rewritten.
## Denpendcies
- Python  3.8.2 (Recommend to use Anaconda)
- Pytorch 2.0
- NVIDIA GPU + CUDA
- Python packages: pip install numpy scipy h5py torchsummary
## Code
Training and testing codes are in the current folder. Run the file run.py to start!
- The code for training is in main_xxx.py (three stages), while the code for testing test.py.
- For training, you need to set the file_path in the main function, adopt t your train set, validate set, and test set as well. Our code train the .h5 file, you may change it through changing the code in main function.
- RSP and SDE stages should be trained before FUG stage, while they can be trained simutaneously. 
- As for testing, you need to set the path in both main and test function to open and load the file.
- A trained model for a single WV3 full resolution image is attached along the code.

