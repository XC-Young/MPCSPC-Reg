## A novel method for registration of MLS and stereo reconstructed point clouds

In this paper, we present a new algorithm for cross-source point cloud registration between MLS point clouds and stereo-reconstructed point clouds. 
To improve the registration performance in this task, our method has two key designs. Firstly, gravity prior is utilized to highlight features in specific directions to narrow the matching pair search and enhance the robustness and efficiency of rotation-equivariant descriptor construction. Secondly, to adapt to noise patterns of stereo-reconstructed point clouds, a novel disparity-weighted hypothesis scoring strategy is proposed to strengthen RANSAC-based transformation estimation. 
We create two new cross-source point cloud registration datasets to evaluate cross-source registration algorithms. The proposed method achieves state-of-the-art performance with a $43.5\%$ higher registration recall on cross-source datasets and a $10\times \sim 70\times$ speedup faster than RANSAC-based baselines. 



## Requirements

Here we offer the FCGF backbone YOHO. Thus FCGF requirements need to be met:

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

Specifically, The code has been tested with:

- Ubuntu 20.04, CUDA 11.1, python 3.7.10, Pytorch 1.7.1, GeForce RTX 3080Ti.

## Installation

- First, create the conda environment:

  ```
  conda create -n msreg python=3.7
  conda activate msreg
  ```

- Second, intall Pytorch. We have checked version 1.7.1 and other versions can be referred to [Official Set](https://pytorch.org/get-started/previous-versions/).

  ```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.1 -c pytorch
  ```

- Third, install MinkowskiEngine for FCGF feature extraction, here we offer two ways according to [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine.git) by using the version we offered:

  ```
  cd MinkowskiEngine
  conda install openblas-devel -c anaconda
  export CUDA_HOME=/usr/local/cuda-11.1
  python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
  cd ..
  ```

  Or following official command installation:

  ```
  pip install git+https://github.com/NVIDIA/MinkowskiEngine.git
  ```

- Fourth, install other packages:

  ```
  pip install -r requirements.txt
  ```

- Finally, compile the [CUDA based KNN searcher](https://github.com/vincentfpgarcia/kNN-CUDA):
  ```
  cd knn_search/
  export CUDA_HOME=/usr/local/cuda-11.1
  python setup.py build_ext --inplace
  cd ..
  ```

## Dataset & Pretrained model

The datasets and pretrained weights have been uploaded to Google Cloud:

For training:
- [KITTI_train](https://drive.google.com/file/d/1mfnGL8pRvc6Rw6m6YnvNKdbpGxGJ081G/view?usp=sharing);

For testing:
- [CTCS](https://drive.google.com/file/d/1UzGBPce5VspD2YIj7zWrrJYjsImSEc-5/view?usp=sharing);
- [CS](https://drive.google.com/file/d/1hyurp5EOzvWGFB0kOl5Qylx1xGelpxaQ/view?usp=sharing);

Pretained model:
- [Pretrained Weights](https://drive.google.com/file/d/1J-nkut2A66fyOQu0B0723yCfRiJLSG4O/view?usp=sharing). (Already added to the main branch.)


Datasets above contain the point clouds (.ply) and keypoints (.txt, 5000 per point cloud) files. Please place the data to `./data/origin_data` following the example data structure as:

```
data/
├── origin_data/
    ├── 3dmatch/
    	└── kitchen/
            ├── PointCloud/
            	├── cloud_bin_0.ply
            	├── gt.log
            	└── gt.info
            └── Keypoints/
            	└── cloud_bin_0Keypoints.txt
    ├── 3dmatch_train/
    ├── ETH/
    └── WHU-TLS/
```

Pretrained weights we offer include FCGF Backbone, Part I and Part II. Which have been added to the main branch and organized following the structure as:

```
model/
├── Backbone/
	└── best_bal_checkpoint.pth
├── PartI_train/
	└── model_best.pth
└── PartII_train/
	└── model_best.pth
```

## Train

To train YOHO, the group input of train set should be prepared using the FCGF model we offer, which is trained with rotation argument in [0,50] deg, by command:

```
python YOHO_trainset.py
```



The training of YOHO is two-stage, you can run which with the commands sequentially:

```
python Train.py --Part PartI
python Train.py --Part PartII
```

## Demo

With the Pretrained/self-trained models, you can try YOHO with:

```
python YOHO_testset.py --dataset demo
python Demo.py
```

## Test on the 3DMatch and 3DLoMatch

To evalute YOHO on 3DMatch and 3DLoMatch:

- Prepare the testset:
  ```
  python YOHO_testset.py --dataset 3dmatch --voxel_size 0.025
  ```
- Evaluate the results:
  ```
  python Test.py --Part PartI  --max_iter 1000 --dataset 3dmatch    #YOHO-C on 3DMatch
  python Test.py --Part PartI  --max_iter 1000 --dataset 3dLomatch  #YOHO-C on 3DLoMatch
  python Test.py --Part PartII --max_iter 1000 --dataset 3dmatch    #YOHO-O on 3DMatch
  python Test.py --Part PartII --max_iter 1000 --dataset 3dLomatch  #YOHO-O on 3DLoMatch
  ```
  Where PartI is YOHO-C and PartII is YOHO-O, max_iter is the ransac times, PartI should be run first. All results will be stored in `./data/YOHO_FCGF`.

## Generalize to the ETH dataset

The generalization results on the outdoor ETH dataset can be got as follows:

- Prepare the testset:

  ```
  python YOHO_testset.py --dataset ETH --voxel_size 0.15
  ```

  If out of memory, you can

  - Change the parameter `batch_size` in `YOHO_testset.py-->batch_feature_extraction()-->loader` from 4 to 1
  - Carry out the command scene by scene by controlling the scene processed now in `utils/dataset.py-->get_dataset_name()-->if name==ETH`

- Evaluate the results:
  ```
  python Test.py --Part PartI  --max_iter 1000 --dataset ETH --ransac_d 0.2 --tau_2 0.2 --tau_3 0.5 #YOHO-C on ETH
  python Test.py --Part PartII --max_iter 1000 --dataset ETH --ransac_d 0.2 --tau_2 0.2 --tau_3 0.5 #YOHO-O on ETH
  ```
  All the results will be placed to `./data/YOHO_FCGF`.

## Generalize to the WHU-TLS dataset

The generalization results on the outdoor WHU-TLS dataset can be got as follows:

- Prepare the testset:

  ```
  python YOHO_testset.py --dataset WHU-TLS --voxel_size 0.8
  ```

- Evaluate the results:
  ```
  python Test.py --Part PartI  --max_iter 1000 --dataset WHU-TLS --ransac_d 1 --tau_2 0.5 --tau_3 1 #YOHO-C on WHU-TLS
  python Test.py --Part PartII --max_iter 1000 --dataset WHU-TLS --ransac_d 1 --tau_2 0.5 --tau_3 1 #YOHO-O on WHU-TLS
  ```
  All the results will be placed to `./data/YOHO_FCGF`.

## Generalize to the KITTI dataset

```
python Test.py --Part PartI --max_iter 1000 --dataset kitti --ransac_d 0.5 --tau_2 2 #YOHO-C on KITTI
python Test.py --Part PartII --max_iter 1000 --dataset kitti --ransac_d 0.5 --tau_2 2 #YOHO-O on KITTI

python Test.py --Part PartI --max_iter 1000 --dataset kittistereo --ransac_d 1.5 --tau_2 2 #YOHO-C on KITTI_Stereo
python Test.py --Part PartII --max_iter 1000 --dataset kittistereo --ransac_d 1.5 --tau_2 2 #YOHO-O on KITTI_Stereo

python Test.py --Part PartI --max_iter 1000 --dataset kittisubseq --ransac_d 1.5 --tau_2 2 #YOHO-C on KITTI_Sub Sequence
python Test.py --Part PartII --max_iter 1000 --dataset kittisubseq --ransac_d 1.5 --tau_2 2 #YOHO-O on KITTI_Sub Sequence

python Test.py --Part PartI --max_iter 1000 --dataset kittiscst --ransac_d 1.5 --tau_2 2 #YOHO-C on KITTI_Scst
python Test.py --Part PartII --max_iter 1000 --dataset kittiscst --ransac_d 1.5 --tau_2 2 #YOHO-O on KITTI_Scst

python Test.py --Part PartI --max_iter 1000 --dataset kitti+360 --ransac_d 1.5 --tau_2 2 
python Test.py --Part PartII --max_iter 1000 --dataset kitti+360 --ransac_d 1.5 --tau_2 2 
```

## Customize YOHO according to your needs

To test YOHO on other datasets, or to implement YOHO using other backbones according to your needs, please refer to `./others/Readme.md`

## Related Projects

We sincerely thank the excellent projects:

- [EMVN](http://github.com/daniilidis-group/emvn) for the group details;
- [FCGF](https://github.com/chrischoy/FCGF) for the backbone;
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) for the 3DMatch dataset;
- [Predator](https://github.com/overlappredator/OverlapPredator) for the 3DLoMatch dataset;
- [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) for the ETH dataset;
- [WHU-TLS](https://www.sciencedirect.com/science/article/pii/S0924271620300836) for the WHU-TLS dataset;
- [PerfectMatch](https://github.com/zgojcic/3DSmoothNet) for organizing the 3DMatch and ETH dataset.
