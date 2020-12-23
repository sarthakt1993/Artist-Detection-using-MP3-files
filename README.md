# Artist Detection using MP3 files
 
## Problem Statement 
The aim of this project is to make a classification architecture which will be used on a dataset to classify artists using their songs. 

## Dataset 
The dataset which we will be using for our project is the **artist20** dataset. This dataset was created by the **Laboratory for the Recognition and Organization of Speech and Audio (LabROSA)** at Columbia University to be used to experiment and evaluate classification performance. 

## Evaluation 
We shall be using F1 score to evaluate our model performance. This is because the sample size for each artist is not same, hence there will class imbalance issues. To counteract that effect, we will be using F1 score as our evaluation metric. 

## Language 
- Python
- Pytorch

## Architectures Implemneted 
- CRNN with Inception Block 
- CRNN with ResNet50 Block  

## Files
- **Project Report** - This was Report was part of the project submission
- **ANN_Project_Main** - IPython notebook with the entire code with comments 
- **util** - File with all the code
