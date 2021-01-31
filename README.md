# Skyline Extraction using Shallow LearningComplete implementation for our paper listed below, the two main components of the code depend upon Python and Matlab.  **[Resource Efficient Mountainous Skyline Extraction using Shallow Learning](https://drive.google.com/file/d/1E7ebOEcuA8FNmh73qaemEjxmA45_QC5o/view)**, [IJCNN2021](https://www.ijcnn.org/)Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Ebrahim Emami](https://scholar.google.com/citations?user=FVQqg0wAAAAJ&hl=en),  [Martin Čadík](http://cadik.posvete.cz), and [George Bebis](https://www.cse.unr.edu/~bebis/) ## RequirementsThe code is straight forward and depends on Python and OpenCV. Have been tested in [conda](https://www.anaconda.com/distribution/) virtualenvironment with Python 3.6.10 and OpenCV 4.3.0 ## DatasetsWe have learned our filter banks based on three dataset i.e., Basalt, Web and CH1 and additionally also tested on GeoPose3K dataset. The first three datasets can be downloaded from here and should be placed in the main directory. The original CH1 dataset is available from authors' [webpage](http://cvg.ethz.ch/research/mountain-localization/). The version provided with this code is just for convenience, please consult original copyrights and terms of usage of CH1 dataset. Additionally, please download GeoPose3K from respective [webpage](http://cphoto.fit.vutbr.cz/geoPose3K/). The GeoPose3K dataset should be placed in the data directory. For reference, this how the directory structure looks like for us.         ```data├── Basalt│     ├── ground_truth│     ├── images   ├── web_dataset│     ├── ground_truth│     ├── images├──         ```## TrainingThe filter banks can be learned using any of the three datasets (i.e., Basalt, Web and CH1). Script for training is in train.py where the filter size, number of bins for structure tensor components (coherence, orientation and strength) and the dataset being used can be adjusted. The gram matrices and filter banks for specific filter size and dataset are saved in the filterBank directory. A specific filter bank then can be used for inference. Run the training:     ```shellpython train.py```## InferenceOnce the filter bank is learned for a specific dataset (e.g., Web), it can then be used to run the inference for any other dataset (e.g., Basalat).The inference code is available in test.py where learned filter bank and test dataset can be specified. The code provides the option to saveintermediate outputs for visualization and save the .mat files which are to be later used in dynamic programming step to get the shortestpath in the multi-stage graph which conforms to the detected skyline for the image. Run the inference:       ```shellpython test.py```## CitationIf you find this work useful, please consider citing our paper:```@inproceedings{ahmad2021resource,  title={Resource Efficient Mountainous Skyline Extraction using Shallow Learning},  author={Ahmad, Touqeer and Emami, Ebrahim and Čadík, Martin and Bebis, George},  booktitle={Proceedings of the 2021 IEEE International Joint Conference on Neural Networks},  year={2021}}```