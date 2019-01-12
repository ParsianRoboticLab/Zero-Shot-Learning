# Parsian Zero-Shot Learning!

This is Parsian Project for ZJU AI Challenge Zero-Shot Learning Competition 2018.

# Files


```
project
│   README.md
└───data
│        └───External
│        │ 
│        └───DatasetA
│        │          └───train 	
│        │          └───test
│ 	     │ 	
│        └───DatasetB
│                   └───train 	
│                   └───test
└───code
│      │   main.py
│
└───submit
	     │   submit_20180926_040506.txt

```

## Dependencies
```
OS: Ubuntu 16.04 LTS [TESTED]
Python Version: 3.7
```

### Python Packages
	1. torch=0.4.1
	2. tourchvision=0.2.1
	3. numpy=1.15.1
	4. scikit-learn=0.19.2
	5. scipy=1.1.0
	6. pillow=5.2.0
	7. matplotlib=3.0.0
	8. tensorflow=1.12.0rc2
	9. opencv-python=3.4.3.18

### GPU CUDA Compiler
```
NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Nov__3_21:07:56_CDT_2017
Cuda compilation tools, release 9.1, V9.1.85
```


> There's no additional **binary** file is needed.



## Usage

1. Put train and test data images in the `data` directory as mentioned in readme.md structure
2. Execute `main.py` to train the nn-model with provided data.

## Authors

[Hamed Hosseini](hellihdhs@gmail.com)
[Alireza Zolanvari](https://github.com/AlirezaZolanvari)
[Amir Mohammad Naderi](#)
[Mohammad Mahdi Shirazi](https://github.com/mhmmdshirazi)
[Mohammad Mahdi Rahimi](https://github.com/Mahi97)

## Description

### GAN and Data Generation

Due to the lack of data for unseen lables the state-of-art for this project was training a Generative Adversarial Network in order to generate the unseen dataset.

According to the GAN Architecture that shows below, the generative network make random image with fake label `ZJULFAKE`. Both discriminator and generator networks will train through the response of the discriminator network to the augmunted dataset including both real and fake data.

![alt text][GAN]

We keep training these two network till the state that the discriminator network can not discriminate the fake and real data, this shows the perfection of the generator network.

At this point we use the trained generator network in the main architecture of our learning system.

Now the problem simplified to a classic classification problem the we have enough data of all labels and we will train our CNN by using augumented dataset from both `Train Data` and `Generated Data`.

![alt text][TRAIN]

Using trained CNN on test data and the result will be a `lable probability vector` for each image. 
(This vector show the membership probability of an image for each label.)

![alt text][TEST]


In the next step we get `weighted average of attribute` for each image, where membership probability of each lable will be the attributes weight for that label. Result attribue vector will be construct by weighted average of each attribute in labels.

Now, we have a attribute vector for each image and find the match label for that attribute vector by measuring the distance of that vector on manifold that construct by attributes per label. The nearest class to image vector will be decided as the correct class for that image. 

![alt text][ALL]

### CNN - RESNET152

To prevent vanishing gradient, we used RESNET152. RESNET use residual connections over a layer.

![alt text][RESNET]


### Manifold
manifold and clustring usage for find the best match label after computing the attribiutes values
![alt text][3d]
![alt text][t-SNE]
![alt text][MDS]
![alt text][Spectral]



[GAN]: ./img/GAN.png "The GAN Arcitecture"
[TEST]: ./img/Test.png "The Test Arcitecture"
[TRAIN]: ./img/Train.png "The Train Arcitecture"
[ALL]: ./img/All.png "The Main Architecture"
[RESNET]: ./img/RESNET.png "The RESNET Architecture"
[3d]: ./img/3d.png "Clustring"
[t-SNE]: ./img/t-SNE.png "t-SNE"
[MDS]: ./img/MDS.png "MDS"
[Spectral]: ./img/Spectral.png "Spectrul"

