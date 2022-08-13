# Using data-processor.py for dataset preparation

`data-processor.py` is a small utility written to transform datasets from raw images to tensors, so that the latter can be imported into a `torch.utils.data.DataLoader` class. If you would like to reimport the [Kaggle dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) used in the original model training or import your custom dataset, you can use this utility to do so. 

**Note:** you can specify any other *path* to your data by using `--path` argument. `data-processor.py` crawls through each subdirectory of the *path* you have provided, which can lead to unintended results (like random images being swept up into the dataset, or wrong imports).

In order to import images using the `data-processor.py`:

1. add a new folder here (you can name it any way you like, the converted dastaset file will be named `dataset-your-name.pt`);
2. put all your images in that folder (note that only `.jpg` files are being supported - you can modify the file to accommodate your desired file format);
3. type `python3 data-processor.py` to launch the utility in default mode (you can specify arguments, for full infomration type `--help`).

### References used during development:
[Link 1 - Easy PyTorch to Load Image and Volume from Folder](https://inside-machinelearning.com/en/easy-pytorch-to-load-image-and-volume-from-folder/)
[Link 2 - DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)