# Skyline Extraction using Shallow Learning

Complete implementation for our paper listed below, the two main components of the code depend upon Python and Matlab.  

**[Resource Efficient Mountainous Skyline Extraction using Shallow Learning](https://drive.google.com/file/d/1k34AX0oiVR0HRmt16UZHE7bW16hz_jg-/view)**, [IJCNN2021](https://www.ijcnn.org/)

Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Ebrahim Emami](https://scholar.google.com/citations?user=FVQqg0wAAAAJ&hl=en),  [Martin Čadík](http://cadik.posvete.cz), and [George Bebis](https://www.cse.unr.edu/~bebis/) 


## Requirements
The shallow learning part of the code depends on Python and OpenCV. It has been tested in [conda](https://www.anaconda.com/distribution/) virtual
environment with Python 3.6.10 and OpenCV 4.3.0. Whereas the dynamic programming part of the code depends on Matlab and has been tested
using Matlab 2016.  


## Datasets
We have learned our filter banks based on three dataset i.e., Basalt, Web and CH1 and additionally also tested on GeoPose3K dataset. The first 
three datasets can be downloaded from [here](https://drive.google.com/file/d/1SVu7fgI7kOcwQgJxGlm7TeiIEBCqYLXu/view?usp=sharing) and should be placed in the main directory. The original CH1 dataset is available from 
authors' [webpage](http://cvg.ethz.ch/research/mountain-localization/). The version provided with this code is just for convenience, 
please consult original copyrights and terms of usage of CH1 dataset. Additionally, please download GeoPose3K from respective 
[webpage](http://cphoto.fit.vutbr.cz/geoPose3K/). The GeoPose3K dataset should be placed in the data directory. For reference, this is how 
the directory structure looks like for us.         

```
data
├── Basalt
│     ├── ground_truth
│     ├── images   
├── CH1
│     ├── cvg
│     │    ├── ground_truth
│     │    ├── images    
│     ├── panoramio
│     │    ├── ground_truth
│     │    ├── images    
│     ├── poor_edge_images 
│     │    ├── ground_truth
│     │    ├── images    
├── web_dataset
│     ├── ground_truth
│     ├── images
├── geoPose3K_rescaled 
│     ├── flickr_1637436219_d912602638_2282_84835246@N00
│     ├── flickr_2083964014_49ba3bfe52_2343_21428225@N00
│     ├── ...
        
```


## Shallow Learning: Training
The filter banks can be learned using any of the three datasets (i.e., Basalt, Web and CH1). Script for training is in train.py where the 
filter size, number of bins for structure tensor components (coherence, orientation and strength) and the dataset being used can be adjusted. 
The gram matrices and filter banks for specific filter size and dataset are saved in the filterBank directory. A specific filter bank 
then can be used for inference. To run the training:     

```shell
python train.py
```

## Shallow Learning: Inference
Once the filter bank is learned for a specific dataset (e.g., Web), it can then be used to run the inference for any other dataset (e.g., Basalt).
The inference code is available in test.py where learned filter bank and test dataset can be specified. The code provides the option to save
intermediate outputs for visualization and save the .mat files which are to be later used in dynamic programming step to get the shortest
path in the multi-stage graph which conforms to the detected skyline for the image. To run the inference:       

```shell
python test.py
```

## Dynamic Programming
Details go here ...


## Miscellaneous
- **Horizon_DCSI.py**

    The file Horizon_DCSI.py provides the implementation for paper: [An Edge-Less Approach to Horizon Line Detection](https://drive.google.com/file/d/1E7ebOEcuA8FNmh73qaemEjxmA45_QC5o/view).
The SVM is trained using CH1 datset and weights for learned SVM are available in misc/Horizon_SVM_Classifier_Baatz_CH1.mat. 

- **NamesOf2895Files.mat** 

    For a consistent comparison with an [earlier work](https://drive.google.com/file/d/1xEm7AsqWGe6VR2ZfCbcxb4i3u5DrsX2B/view), we used the same
2895 images from the GeoPose3K dataset. The filenames for these specific images are provided in NamesOf2895Files.mat


## Citation

If you find this work useful, please consider citing our paper:

```
@inproceedings{ahmad2021resource,
  title={Resource Efficient Mountainous Skyline Extraction using Shallow Learning},
  author={Ahmad, Touqeer and Emami, Ebrahim and Čadík, Martin and Bebis, George},
  booktitle={Proceedings of the 2021 IEEE International Joint Conference on Neural Networks},
  year={2021}
}
```
