# Skyline Extraction using Shallow LearningPythonic implementation of our paper: **[Resource Efficient Mountainous Skyline Extraction using Shallow Learning](https://drive.google.com/file/d/1E7ebOEcuA8FNmh73qaemEjxmA45_QC5o/view)**, [IJCNN2021](https://www.ijcnn.org/)Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Ebrahim Emami](https://scholar.google.com/citations?user=FVQqg0wAAAAJ&hl=en),  [Martin Čadík](http://cadik.posvete.cz), and [George Bebis](https://www.cse.unr.edu/~bebis/) ## RequirementsThe code is straight forward and depends on Python and OpenCV. Have been tested in [conda](https://www.anaconda.com/distribution/) virtualenvironment with Python 3.6.10 and OpenCV 4.3.0 ## DatasetsFor training we have worked with three dataset i.e., Basalt, Web and CH1. The datasets can be downloaded from here and should be placedin the main directory. The original CH1 dataset is available from authors' [webpage](http://cvg.ethz.ch/research/mountain-localization/).The version provided with this code is just for convenience, please consult original copyrights and terms of usage of CH1 dataset.      ```data├── Basalt│     ├── ground_truth│     ├── images   ├── web_dataset│     ├── ground_truth│     ├── images├──         ```## TrainingThe filter banks can be learned using any of the three datasets. Script for training is in train.py where the filter size, number of binsfor coherence, orientation and strength and the dataset being used can be adjusted. The filter banks with specific filter size and datasetare saved in the filterBank directory.   ```shellpython train.py```## CitationIf you find this work useful, please consider citing our paper:```@inproceedings{ahmad2021resource,  title={Resource Efficient Mountainous Skyline Extraction using Shallow Learning},  author={Ahmad, Touqeer and Emami, Ebrahim and Čadík, Martin and Bebis, George},  booktitle={Proceedings of the 2021 IEEE International Joint Conference on Neural Networks},  year={2021}}```