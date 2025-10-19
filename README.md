# Background of this project
This project was completed for my Computer Science capstone. 
<br></br>
I found the research paper for the CNN referred to as the original CNN model that inspired me to make a better CNN that worked on larger images and used a smaller more relavant part of the image during training. 
<br></br>
The papers on PReLU and Seperable Convolution layers were used in the design of the new CNN.
<br></br>
<br></br>
# KerasDeepFakeDetection Credits
The original CNN model is based on this paper: https://doi.org/10.1109/ACCESS.2023.3251417
<br></br>
The PReLU function paper is here: https://arxiv.org/pdf/1502.01852
<br></br>
The Seperable Convolution layer paper: https://arxiv.org/pdf/1610.02357/1000
<br></br>
# How to use
Download PyCharm IDE and open the project folder. Configure the project to use the latest version of python 3.11
<br></br>
The good thing is that now you can just use the IDE to pull in the proper dependencies that should work.
<br></br>
<br></br>
Run the MakeCSV.py file: ```python MakeCSV.py```
<br></br>
The MakeCSV.py file will make a folder called "images" in the same directory as MakeCSV.py. The "images" folder will have a folder called "test", "train", and "val". Inside the "test" and "val" folders are more folders called "fake_image" and "real_image". Put the real and fake images you have in their proper folders.
<br></br>
Then run MakeCSV.py again: ```python MakeCSV.py```
<br></br>
This will make a .csv file that contains all of the file names and classes of each image. Run this script everytime you add or remove or edit images in the folders.
<br></br>
<br></br>

Link to original repository: https://github.com/BinaryGears/KerasDeepFakeDetection/
