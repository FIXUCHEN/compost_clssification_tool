First of all, please ensure the Python environment and run the following commands to install the relevant dependencies：
pip install -r requirements.txt


Data： Catalogue of data sets

data:
->apple
->banana
->....
....

roc_curve： The roc curve catalog, containing each class of

datasets.py： Dataset Loader
models.py： model code， VGG16,VGG19 , ResNet18

resnet18_loss.png：loss
resnet18_metric.png： Accuracy curve during training

train_fun.py： Training correlation function
train.py： For the training script, please run the script, note! Please backup the models in saved_models before running to prevent automatic overwriting due to the training
test.py： Test script that generates the relevant image in roc_curve
window.py： Forms to run scripts for uploading, recognizing, etc.

The following two files need to be downloaded using LFS. You can download the file separately from the Github page, or use git lfs clone.
saved_models/vgg16.pth
saved_models/vgg19.pth
