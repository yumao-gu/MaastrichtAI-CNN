# MaastrichtAI-CNN
CNN for emotion recognition using fre2013


#   eviroment
We use python3 with torch 1.5 to build and run our model. 

	pip3 install numpy,pandas,torch,matplotlib,torchvision

before you run the script.


#   data set
We use the fre2013 dataset to train and test our model. These data set includes 7 classes.

	https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge

before run the cnn.py, we should unzip the data in the directory and change the data folder name to 'data'. After that you should run the 
	
	python3 fre2013_process.py

to extract the 'train', 'val' and 'test' images from the csv file. The images are split into 7 folders which means different emotions.

#   train and test
just run 

	python3 cnn.py
