from models import ResNet18 , VGG16 , VGG19
from datasets import ImageDataset
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc , classification_report , confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# augmentation
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

datasets = ImageDataset('data',transform = transform)
class_nums = len(datasets.name)

# choose model这里选择要比较的模型
model_names = ['resnet18','vgg16','vgg19']

if __name__ == '__main__':
    BATCH_SIZE = 32

    os.makedirs('roc_curve',exist_ok = True)
    os.makedirs('other_output',exist_ok = True)    

    train_size , valid_size , test_size = int(len(datasets)*0.8),int(len(datasets)*0.1),len(datasets)-int(len(datasets)*0.8)-int(len(datasets)*0.1)
    _,_,test_datasets = torch.utils.data.random_split(datasets,[train_size,valid_size,test_size],generator = torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_datasets,batch_size = BATCH_SIZE,shuffle = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    roc_all = []
    super_roc_all = []
    for model_name in model_names:
        if model_name == 'resnet18':
            model = ResNet18(class_nums)
            model.load_state_dict(torch.load('saved_models/resnet18.pth',map_location = device))
        elif model_name == 'vgg16':
            model = VGG16(class_nums)
            model.load_state_dict(torch.load('saved_models/vgg16.pth',map_location = device))
        elif model_name == 'vgg19':
            model = VGG19(class_nums)
            model.load_state_dict(torch.load('saved_models/vgg19.pth',map_location = device))
        
        model.to(device)
        model.eval()

        category = ['Carambola', 'Pitaya', 'apple', 'banana', 'cardboard', 'compost', 'glass', 'kiwi', 'mango', 'metal', 'orange', 'paper', 'peach', 'plastic', 'tamotoes', 'trash']

        super_category = {'pos':['Carambola', 'Pitaya', 'apple', 'banana', 'kiwi', 'mango', 'orange', 'peach', 'tamotoes', 'compost'],
                        'neg':['glass','metal','plastic','cardboard','paper','trash']
                        }
        super_category['pos'] = [category.index(i) for i in super_category['pos']]
        super_category['neg'] = [category.index(i) for i in super_category['neg']]

        # compute tp , tn , fp , fn for each super category ,and each category
        y_true = []
        y_pred = []
        pred_score = []
        with torch.no_grad():
            
            for batch in test_loader:
                images , labels = batch
                images , labels = images.to(device) , labels.to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs,dim = -1)
                score,preds = torch.max(outputs,1)
                id_preds = preds.cpu().numpy()
                id_labels = labels.cpu().numpy()
                pred_score.extend(score.cpu().numpy())

                y_true.extend(id_labels)
                y_pred.extend(id_preds)
            
        # show classification report and save to csv file
        report = classification_report(y_true,y_pred,target_names = category,output_dict = True)
        report = pd.DataFrame(report).transpose()
        report.to_csv('other_output/classification_report_%s.csv'%model_name)

        # show confusion matrix and plot it,add number to each cell
        cm = confusion_matrix(y_true,y_pred)
        cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
        plt.figure(figsize = (10,10))
        plt.imshow(cm,interpolation = 'nearest',cmap = plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(category))
        plt.xticks(tick_marks,category,rotation = 45)
        plt.yticks(tick_marks,category)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j,i,int(cm[i,j]*100)/100,
                        horizontalalignment = 'center',
                        color = 'white' if cm[i,j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig('other_output/confusion_matrix_%s.png'%model_name)
        plt.close()


        
        # show roc curve for each category
        path = 'roc_curve'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        y_label = label_binarize(y_true,classes = list(range(class_nums)))
        for i in range(class_nums):
            fpr[i],tpr[i],_ = roc_curve(y_label[:,i],pred_score)
            roc_auc[i] = auc(fpr[i],tpr[i])
            plt.figure()
            plt.plot(fpr[i],tpr[i],label = 'ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0,1],[0,1],'k--')
            plt.xlim([0.0,1.0])
            plt.ylim([0.0,1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(category[i])
            plt.legend(loc = 'lower right')
            plt.savefig(path+'/'+category[i]+'_%s.png'%model_name)
            plt.close()
        
        
        # convert y_true and y_pred to super category
        y_true = [1 if i in super_category['pos'] else 0 for i in y_true]
        y_pred = [1 if i in super_category['pos'] else 0 for i in y_pred]

        # show roc curve for super category
        fpr['pos'],tpr['pos'],_ = roc_curve(y_true,pred_score)
        roc_auc['pos'] = auc(fpr['pos'],tpr['pos'])
        plt.figure()
        plt.plot(fpr['pos'],tpr['pos'],label = 'ROC curve (area = %0.2f)' % roc_auc['pos'])
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Compostable or not')
        plt.legend(loc = 'lower right')
        plt.savefig(path+'/'+'Compostable_%s.png'%model_name)
        plt.close()

        roc_all.append([fpr,tpr,roc_auc])

        # show report for super category and save to csv file
        report = classification_report(y_true,y_pred,target_names = ['neg','pos'],output_dict = True)
        report = pd.DataFrame(report).transpose()
        report.to_csv('other_output/super_classification_report_%s.csv'%model_name)

        # show confusion matrix for super category and plot it,add number to each cell
        cm = confusion_matrix(y_true,y_pred)
        cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
        plt.figure(figsize = (10,10))
        plt.imshow(cm,interpolation = 'nearest',cmap = plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks,['neg','pos'],rotation = 45)
        plt.yticks(tick_marks,['neg','pos'])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j,i,int(cm[i,j]*100)/100,
                        horizontalalignment = 'center',
                        color = 'white' if cm[i,j] > thresh else 'black')

        plt.tight_layout()
        plt.savefig('other_output/super_confusion_matrix_%s.png'%model_name)
        plt.close()

    # compare roc curve for different models
    
    for i in range(len(category)):
        for j in range(len(model_names)):
            plt.plot(roc_all[j][0][i],roc_all[j][1][i],label = model_names[j]+' (area = %0.2f)' % roc_all[j][2][i])
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0.0,1.0]) 
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(category[i])
        plt.legend(loc = 'lower right')
        plt.savefig('roc_curve/'+category[i]+'_compare.png')
        plt.close()
    
    for j in range(len(model_names)):
        plt.plot(roc_all[j][0]['pos'],roc_all[j][1]['pos'],label = model_names[j]+' (area = %0.2f)' % roc_all[j][2]['pos'])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Compostable or not')
    plt.legend(loc = 'lower right')
    plt.savefig('roc_curve/'+'Compostable_compare.png')
    plt.close()