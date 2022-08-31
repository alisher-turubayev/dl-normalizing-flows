# Utilizing Normalizing Flows for Anime Face Generation - Code
### Deep Learning Summer 2022 - Final Project
### Alisher Turubayev, Hasso-Platter Institute - M.Sc. in Digital Health student

This is the GitHub repository for the code part of the Deep Learning final project on normalizing flows and their performance in generating novel anime faces. 

**Used dataset:**

[Kaggle - Anime Faces Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

**Used algorithms:**

Density estimation using RealNVP by Dihn, L., Sohl-Dickstein, J. and Bengio, S. (2017) [arXiv: 1605.08803](https://arxiv.org/pdf/1605.08803.pdf)

Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks by Radford, A., Metz, L. and Chintala, S. (2016) [arXiv: 1511.06434](https://arxiv.org/pdf/1511.06434.pdf)

**Code references and used repositories:**

~~[Glow implementation on Pytorch by Joost van Amersfoort (@y0ast) & Caleb Carr (@calebmcarr)](https://github.com/y0ast/Glow-PyTorch)~~

*Removed due to consistent issues / concerns about implementation. However, the code was used during training of a Glow algorithm, and the results from that experiment were included in the final report - see [glow_progression.jpg](samples/glow_progression.jpg).*

[Implementation of selected normalizing flows on PyTorch by Ilya Kostrikov (@ikostrikov)](https://github.com/ikostrikov/pytorch-flows)

[realNVP by Fangzhou Mu (@fmu2)](https://github.com/fmu2/realNVP)

[DCGAN by Nathan Inkawhich @inkawhich](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py)

## Results

Results are reported in the [final report](report/final_report.pdf) and the [code demo](etc/colab_demo.ipynb). Samples from the training are available in the `samples` folder. 

## Reproducing Project Results

To reproduce project results, consult [code demo](etc/colab_demo.ipynb) file. 

### For RealNVP:

*PATH* refers to the dataset path - see an additional [DATA_README](datasets/DATA_README.md) for information on how to correctly import a dataset and about data prunning used. *OUTPUT_PATH* refers to the path of program outputs - by default, states are saved for all trained models as either `realnvp_state{_optim}.pt` or `{generator,discriminator}_state{_optim}.pt`.  

For random-seed, 64 x 64 x 3 -> 4 x 4 x 48 training with 4 residual blocks/32 features per block:
```
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --nofixed
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --nofixed
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --nofixed
```
*Note:* due to the limitations of Google Colab free tier, the author has trained the models in several passes, which are reflected in the commands above.

For fixed seed, 64 x 64 x 3 -> 4 x 4 x 48 training with 4 residual blocks/32 features per block:
```
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --fixed-seed 409
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --fixed-seed 409
python3 main.py --algo realnvp --epochs 25 --res-blocks 4 --base-dim 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --fixed-seed 409
```

For training 32 x 32 x 3 -> 16 x 16 x 6, 8 residual blocks/64 features per block, download files from commit [bab504a](https://github.com/alisher-turubayev/dl-normalizing-flows/commit/bab504a2671de6ae5e3032e4d0fbb661d1ae563c), and use the following command:
```
python3 main.py --algo realnvp --epochs 50 --image-size 32 --datapath PATH --output-dir OUTPUT_PATH --nofixed
python3 main.py --algo realnvp --epochs 50 --image-size 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --nofixed
python3 main.py --algo realnvp --epochs 50 --image-size 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --nofixed
python3 main.py --algo realnvp --epochs 20 --image-size 32 --datapath PATH --output-dir OUTPUT_PATH --nofresh --saved-path OUTPUT_PATH/states --nofixed
```

### For DCGAN:

```
!python3 dl-normalizing-flows/main.py --algo gan --epochs 500 --datapath PATH --output-dir OUTPUT_PATH --nofixed
```

## General Usage Information

For general usage, see `-h` or `--help`. Additional information is available in additional [DATA_README](datasets/DATA_README.md) file.