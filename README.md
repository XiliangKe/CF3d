# CF3d
**CF3d: Category Fused 3D Point Cloud Retrieval.** 
Zongyi Xu, Zuo Li, Yilong Chen.

## Introduction
3D point cloud retrieval technology that facilitates resource reuse has become a hot research topic in the field of computer vision. In recent years, many view-based retrieval methods have been proposed. Despite achieving state-of-the-art performance in many benchmarks, these methods inevitably lose a large amount of spatial information due to the nature of view projection process. In this paper, we propose a category fused retrieval method that directly extracts geometric and semantic feature from the 3D point cloud. Specifically, we incorporate category information by learning a separate network for point cloud classification. Apart from the conventional cross entropy loss, we design
an intra-class constrained loss function to make the intraclass features more compact. We design an offset-attention module with an implicit Laplacian operator to reduce the noise in our feature learning process. In addition, we devise a
data-driven 3D augmentation module that learns to generate difficult but meaningful examples for model training. Consistency loss is added to ensure that the augmented sample lies closely to its counterpart in the feature space. Extensive experiments are conducted on ModelNet40 and ShapeNetPart datasets to demonstrate that our method outperforms four state-of-the-art methods.


## Installation
Please install [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [sklearn](https://scikit-learn.org/).
The code has been tested with Python 3.7, pytorch 1.2, CUDA 10.0 and cuDNN 7.6 on Ubuntu 16.04.

## Usage
### Point Cloud Retrieval

Download the ModelNet40 dataset and ShapeNet Part dataset. 

To train the model,
```
python train_PA.py --data_dir Dataset_Folder
```

To evaluate the model,
```
python eval_PA.py --checkpoint /path/to/checkpoint 
```

## License
This repository is released under MIT License (see LICENSE file for details).



