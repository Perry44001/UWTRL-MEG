# Underwater Target Recognition (Acoustic Recognition) Project
This project focuses on the field of underwater target recognition, utilizing acoustic recognition technology to achieve classification and localization of underwater targets. Through carefully designed model architectures, data processing workflows, and training strategies, it aims to improve the accuracy and efficiency of underwater target recognition.

The paper has been published in the MDPI journal *Remote Sensing*. For detailed information, please refer to: [https://www.mdpi.com/3465976](https://www.mdpi.com/3465976)

## 1. Project Overview
The core of the project is a deep learning-based underwater target recognition system, which mainly includes modules such as data processing, model construction, training, and evaluation. The data processing module is responsible for extracting acoustic features from audio files, such as mel spectrograms, Welch power spectra, and average amplitude spectra, and constructing datasets. The model construction covers multiple neural network models, such as `class_network` for classification, `local_network` for localization, and the multi-task model `MultiTaskLossWrapper` that combines both. The training and evaluation module uses optimization algorithms and evaluation metrics to train and evaluate the model performance.

Citation
```md
# MDPI and ACS Style
Qian, P.; Wang, J.; Liu, Y.; Chen, Y.; Wang, P.; Deng, Y.; Xiao, P.; Li, Z. Multi-Task Mixture-of-Experts Model for Underwater Target Localization and Recognition. Remote Sens. 2025, 17, 2961. https://doi.org/10.3390/rs17172961

# AMA Style
Qian P, Wang J, Liu Y, Chen Y, Wang P, Deng Y, Xiao P, Li Z. Multi-Task Mixture-of-Experts Model for Underwater Target Localization and Recognition. Remote Sensing. 2025; 17(17):2961. https://doi.org/10.3390/rs17172961

# Chicago/Turabian Style
Qian, Peng, Jingyi Wang, Yining Liu, Yingxuan Chen, Pengjiu Wang, Yanfa Deng, Peng Xiao, and Zhenglin Li. 2025. "Multi-Task Mixture-of-Experts Model for Underwater Target Localization and Recognition" Remote Sensing 17, no. 17: 2961. https://doi.org/10.3390/rs17172961

# APA Style
Qian, P., Wang, J., Liu, Y., Chen, Y., Wang, P., Deng, Y., Xiao, P., & Li, Z. (2025). Multi-Task Mixture-of-Experts Model for Underwater Target Localization and Recognition. Remote Sensing, 17(17), 2961. https://doi.org/10.3390/rs17172961
```

Access to Dataset, Network Code, and Training Weight Files
The following are download links for underwater acoustic target radiation noise datasets, network code, and training weight files (all accessed on August 24, 2025). If a link is inaccessible, you can use the alternative link provided after "or" — the alternative link has been verified to have no regional access restrictions.
1. Dataset Download
Main link: https://modelscope.cn/datasets/qianpeng897/DS3500
or Alternative link: https://huggingface.co/datasets/peng7554/DS3500
2. Network Code Acquisition
Main link: https://gitee.com/open-ocean/UWTRL-MEG
or Alternative link: https://github.com/Perry44001/UWTRL-MEG
3. Training Weight File Download
Main link: https://modelscope.cn/models/qianpeng897/UWTRL-MEG
or Alternative link: https://huggingface.co/peng7554/UWTRL-MEG

## 2. Project Structure
```
project/
│
├── nw_class_moe.py     # Definition of classification network model with MoE structure
├── nw_local_moe.py     # Definition of localization network model with MoE structure
├── nw_mtl.py           # Definition of multi-task learning model
├── md_moe_rl.py        # Audio data processing and dataset construction
├── train_mtl.py        # Main program for model training and testing
├── pltpdf.py           # Visualization of training process data
├── requirements.txt    # List of project dependency libraries
└── README.md           # Project description document
├── train.bat           # Training script for batch experiments

├── analyze_plt/        # Scripts for model performance analysis and visualization
│   ├── analyze_localization.py         # Localization performance analysis and visualization
│   ├── analyze_training_time.py        # Training time analysis and visualization
│   ├── plot_params_vs_acc.py           # Visualization of relationship between model parameters and accuracy
│   ├── plot_train_params.py            # Visualization of training parameters
│   ├── plot_training_cost_vs_acc.py    # Visualization of relationship between training cost and accuracy
│   ├── print_network_size.py           # Calculation of network model size
│   ├── params.csv                      # Data of model parameters and accuracy
│   └── train_time2.csv                 # Training time data
│
├── data_gen/           # Files related to underwater acoustic channel data generation
│   ├── dataOcnMptEhc_Pos.m     # Main process for marine acoustic channel data generation
│   ├── Pos1Azi1freq100Hz.env   # BELLHOP-related environment parameter file
│   ├── Pos1Azi1freq100Hz.brc   # BELLHOP-related configuration file
│   ├── Pos1Azi1freq100Hz.bty   # BELLHOP-related seabed terrain data file
│   ├── Pos1Azi1freq100Hz.ssp   # BELLHOP-related sound speed profile data file
│   ├── Pos1Azi1freq100Hz.arr   # Arrival data file generated by BELLHOP
│   ├── bellhop.m               # Run BELLHOP program
│   ├── depd.m                  # Copy files to current working directory
│   ├── funDirFolder.m          # List all folders under the specified path
│   ├── funNorm.m               # Perform mean normalization on signals
│   ├── funOME.m                # Marine environment multipath effect modeling function
│   ├── funReadTestLb.m         # Read test label text file
│   ├── help.md                 # Contains help information such as exporting Conda environment dependencies
│   ├── kfoldDataSplit.m        # K-fold cross-validation data splitting
│   ├── natsort.m               # Natural sorting of text arrays
│   ├── natsortfiles.m          # Natural sorting of filenames or folder names
│   └── read_arrivals_bin.m     # Read binary arrival data files generated by BELLHOP
```

## 3. Installation Guide
1. **Environment Preparation**: Ensure that a Python environment is installed, and it is recommended to use Python 3.7 or a higher version.
2. **Install Dependencies**: In the project root directory, execute the following command to install the required dependency libraries:
```bash
pip install -r requirements.txt
```

## 4. Data Preparation
1. Prepare files containing underwater target audio data. The audio format should be readable by the `librosa` library, and a sampling rate of 16kHz is recommended.
2. Data files should be arranged line by line, with each line in the format `audio file path\tlabel\tdistance\tdepth`, e.g., `E:/data/audio1.wav\t0\t10.0\t5.0`.
3. Organize training and test data into list files respectively, such as `train_list.txt` and `test_list.txt`, and prepare a category label list file `label_list.txt` with one label per line.
4. The configuration file `config.json` must contain two fields: `Rrmax` and `Szmax`, which are used for normalization of distance and depth, respectively.

Download Open-Source Datasets
Original shipsear data with 16k sampling and 5s segmentation
https://www.doubao.com/drive/s/8623c37953c6a3d6
Shipsear data with 16k sampling and 5s segmentation processed by marine acoustic channel
https://www.doubao.com/drive/s/4914e4ad2b3ec87a

### One-Click Dataset Generation Using Matlab Scripts
#### 1. Code Overview
The Matlab code in the `data_gen` folder is mainly used to generate underwater acoustic channel data and simulate multipath effects in marine environments. These codes include functions such as loading BELLHOP simulation results, applying sound field propagation models to audio data, and generating multi-position acoustic feature datasets.

#### 2. Main Files and Functions
- `dataOcnMptEhc_Pos.m`: Main process for marine acoustic channel data generation, including loading original audio data, applying sound field propagation models, generating multi-position acoustic feature datasets, and saving results to the specified path.
- `funOME.m`: Marine environment multipath effect modeling function, which adds multipath effects to the input time-domain signal.
- `funReadTestLb.m`: Read test label text files and return a cell array containing each line of text.
- `read_arrivals_bin.m`: Read binary arrival data files generated by BELLHOP.
- `bellhop.m`: Run the BELLHOP program.
- `funNorm.m`: Perform mean normalization on signals.
- `depd.m`: Copy files to the current working directory.
- `Pos1Azi1freq100Hz.env`: Configuration file containing environment parameters for BELLHOP simulation.

#### 3. Usage Method
1. Ensure that the BELLHOP program is installed in the Matlab environment, and its executable file `bellhop.exe` is in Matlab's search path.
2. Modify the `oriDataPath` and `ocnDataPath` variables in the `dataOcnMptEhc_Pos.m` file to specify the original audio data path and the storage path for generated data, respectively.
3. Run the `dataOcnMptEhc_Pos.m` file to start generating underwater acoustic channel data.

## 5. Model Training
In the project root directory, use the following command for model training:
```bash
python train_mtl.py --model [model name] --feature [feature name] --task_type [task type] --num_epoch [number of training epochs] --train_list_path [training data list path] --test_list_path [test data list path] --label_list_path [label list path] --num_classes [number of classes] --batch_size [training batch size] --test_batch [test batch size] --lr [learning rate] --weight-decay [weight decay coefficient]
```
Example:
```bash
python train_mtl.py --model meg --feature stft --task_type mtl --num_epoch 150 --train_list_path pathtoyourdataset/train_list.txt --test_list_path pathtoyourdataset/test_list.txt --label_list_path pathtoyourdataset/label_list.txt --num_classes 5 --batch_size 64 --test_batch 64 --lr 0.001 --weight-decay 5e-4
```

Parameter Description:
- `--num_epoch`: Number of training epochs, default 150.
- `--model`: Model name, optional values: `meg`, `mcl`, `swin`, `meg_blc`, `meg_mix`, `resnet18`, `resnet50`, `convnext`, `vgg16`, `vgg19`, `mobilenetv2`, `densenet121`, `swin`, default `meg`.
- `--feature`: Feature name, optional values: `stft`, `mel`, `cqt`, `mfcc`, `gfcc`, default `stft`.
- `--task_type`: Task type, optional values: `mtl`, `classification`, `localization`, default `mtl`.
- `--train_list_path`: Path to the training data list, pointing to a text file containing training data, with each line in the format `audio file path\tlabel\tdistance\tdepth`.
- `--test_list_path`: Path to the test data list, pointing to a text file containing test data, with the same format as the training data list.
- `--label_list_path`: Path to the label list, pointing to a text file containing category labels, with one label per line.
- `--num_classes`: Number of classification categories.
- `--batch_size`: Training batch size, default 64.
- `--test_batch`: Test batch size, default 64.
- `--lr`: Initial learning rate, default 0.001.
- `--weight-decay`: Weight decay coefficient, default 5e-4.

## 6. Model Evaluation
1. During training, the model will automatically evaluate after each training epoch and output metrics such as test accuracy, confusion matrix, and test loss.
2. To perform only evaluation, add the `--evaluate` parameter to the training command:
```bash
python train_mtl.py --evaluate --test_list_path [test data list path] --label_list_path [label list path] --num_classes [number of classes] --batch_size [test batch size]
```

## 7. Visualization Analysis
After training, the `pltpdf.py` script will generate curves such as training loss, multi-task weights, accuracy, and average absolute error (ABSE) based on the data recorded during training, and save them as PDF files. The files are saved in the model save path and can be used to analyze the model training process and performance.

## 8. Code Examples
### 1. Audio Feature Extraction
```python
from md_moe_rl import load_audio

audio_path = "example_audio.wav"
features, welch_spectrum, avg_amp_spectrum = load_audio(audio_path)
print("Mel spectrogram shape:", features.shape)
print("Welch spectrum shape:", welch_spectrum.shape)
print("Average amplitude spectrum shape:", avg_amp_spectrum.shape)
```

### 2. Model Calling
```python
import torch
from nw_mtl import MultiTaskLossWrapper

input_size = 200
channels = 512
embd_dim = 192
num_classes = 5
model = MultiTaskLossWrapper(input_size, channels, embd_dim, num_classes)

# Generate random input data for simulation
batch_size = 16
sequence_length = 100
input_tensor = torch.randn(batch_size, input_size, sequence_length)
welchx = torch.randn(batch_size, input_size)
avgx = torch.randn(batch_size, input_size)
label = torch.randint(0, num_classes, (batch_size,))
Rr = torch.rand(batch_size)
Sz = torch.rand(batch_size)

loss, output, outtaskLocR, outtaskLocD, prec = model(input_tensor, welchx, avgx, label, Rr, Sz)
print("Loss value:", loss.item())
print("Classification output shape:", output.shape)
print("Distance localization output shape:", outtaskLocR.shape)
print("Depth localization output shape:", outtaskLocD.shape)
print("Multi-task weights:", prec)
```

## 9. Contribution Guidelines
1. Developers are welcome to contribute to the project. If you find issues or have improvement suggestions, you can submit Issues in the GitHub repository.
2. To submit code contributions, you need to first fork the project repository, make modifications locally, submit a Pull Request, and describe the modification content and purpose in detail.

## 10. License
This project adopts the MIT License. For details, please refer to the `LICENSE` file in the project root directory.

## 11. Acknowledgments
Thanks to all individuals who provided help and support for the project, as well as the developers of related open-source libraries, whose work laid a solid foundation for the smooth progress of the project.