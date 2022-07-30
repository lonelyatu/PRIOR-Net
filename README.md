# PRIOR-Net & PRIOR
The implementation of PRIOR-Net and PRIOR

# Preparation
The training and test data can be downloaded at https://drive.google.com/drive/folders/1_1Ll5frH72_HwdVmGERrNfnbSpqZqphj.
You will get the train.zip, test.zip and PriorNet.
First, put the "PriorNet" into the current folder "PRIOR-Net".
Next, unzip the train.zip and test.zip to get the folders "train" and "test" and put them in the current directory.


# Training for PRIOR-Net
1. Enter the folder "PRIOR-Net" and run "python TFRecordOp.py" and the files in the "TFRecordsFile" will be saved as tfrecord files

2. run "python main_PriorNet.py" and the model will be saved every epoch

# Testing for PRIOR-Net
1. run "python Test_PriorNet.py" to test the files

# Testing for PRIOR
1. Enter the folder "PRIOR" and run "python main_PRIOR.py". Before run the PRIOR, you should first train the PRIOR-Net or directly use the trained model we provided in the folder "PriorNet" 

# Environment

cuda 10.0

python 3.6.13

TensorFlow 1.15.4

Numpy 1.16.0

Scipy 1.2.1

tigre (https://github.com/CERN/TIGRE)

# Description
Due to the limitation, we only provide one patient to train and one patient to test. You can use your only datasets to train and test.



