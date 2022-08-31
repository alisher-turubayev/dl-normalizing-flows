# Information on Dataset Preparation / Loading

## Prunning

The original Kaggle dataset is ~64k images; however, about ~17k images are smaller than 64x64. As per discussions with the project supervisor, these images were prunned from the dataset, resulting in 47,775 images. 

The script for image prunning written in Bash is available in the [utils folder](https://github.com/alisher-turubayev/dl-normalizing-flows/tree/master/utils) for the project. 

## Dataset Loading

Because the program uses `torch.datasets.ImageFolder` data loader, the file structure is important to ensure that the dataset gets recognized. The author used the following structure:

```
datasets/
    -> kaggle-full/
        -> data/
            -> 0.jpg
            -> 1.jpg
            ...
```

See [torch.datasets.ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder) for additional information.