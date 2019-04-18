# Parsian Zero-Shot Learning!
In this project the goal was recognizing the unseen picture classes using the attributes of seen classes.

This is Parsian Project for ZJU AI Challenge Zero-Shot Learning Competition 2018.

The contest attracted a total of **3225** teams from all over the world to participate in the competition.

My team **“Parsian”** ranked **14th** at the end of the competition.

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

[Mohammad Mahdi Shirazi](https://github.com/mhmmdshirazi)

[Mohammad Mahdi Rahimi](https://github.com/Mahi97)

[Hamed Hosseini](https://github.com/hamed-hosseini)

[Alireza Zolanvari](https://github.com/AlirezaZolanvari)

[Amir Mohammad Naderi](https://github.com/Amiiir)


------

## Description

### GAN and Data Generation

Due to the lack of data for unseen labels, the state-of-art for this project was training a Generative Adversarial Network in order to generate the unseen dataset.

According to the GAN Architecture (Figure bellow), the generative network makes a random image with a fake label `ZJULFAKE`. Both discriminator and generator networks will train through the response of the discriminator network to the augmented data set including both real and fake data.

![alt text][GAN]

Training procedure of both networks continues until the state that the discriminator network cannot discriminate the fake and the real data, this shows the generator network works perfectly. At this point, the trained generator network was used in the main architecture of the learning system.

Now the problem is simplified to a classic classification then the network has enough data of all labels, and the main CNN will be trained using the augmented dataset from both Train Data and Generated Data (Using GAN).

![alt text][TRAIN]

Using trained CNN on test data, and the result will be a **Label Probability** Vector for each image.
(This vector shows the membership probability of an image for each label.)

![alt text][TEST]


In the next step, a **Weighted Average** of Attribute is extracted for each image, where the membership probability of each label will be the attributes weight for that label. Result attribute vector will be constructed by a weighted average of each attribute.

Now, each image has its attribute vector. Hence, finding the matched label for that attribute vector by measuring the distance of that vector on a manifold is possible. In the end, the nearest class to image vector will be selected as the correct class for that image.

![alt text][ALL]

### CNN - RESNET152

To prevent the vanishing gradient, a RESNET152 has been used. Here is the RESNET152 structure.

![alt text][RESNET]


### Manifold
Manifold and clustering usage for find the best match label after computing the attributes values is depicted in the figure bellow.
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

