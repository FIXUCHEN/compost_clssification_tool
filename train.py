from models import ResNet18 , VGG16,VGG19
from datasets import ImageDataset
from train_fun import train
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

datasets = ImageDataset('data',transform = transform)
class_nums = len(datasets.name)


model_name = 'vgg19' # resnet18 , vgg16 , vgg19 ,choose 模型选择
if model_name == 'resnet18':
    model = ResNet18(class_nums)
elif model_name == 'vgg16':
    model = VGG16(class_nums)
elif model_name == 'vgg19':
    model = VGG19(class_nums)
def acc_fun(pred,label):
    pred = torch.argmax(pred,dim = -1)
    acc = torch.sum(pred == label).float() / label.size(0)
    return acc.item()
if __name__ == '__main__':
    
    BATCH_SIZE = 32
    EPOCHS = 30
    # split data set 划分训练集、验证集、测试集
    train_size , valid_size , test_size = int(len(datasets)*0.8),int(len(datasets)*0.1),len(datasets)-int(len(datasets)*0.8)-int(len(datasets)*0.1)
    train_datasets,valid_datasets,test_datasets = torch.utils.data.random_split(datasets,[train_size,valid_size,test_size],generator = torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_datasets,batch_size = BATCH_SIZE,shuffle = True)
    valid_loader = DataLoader(valid_datasets,batch_size = BATCH_SIZE,shuffle = True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    milestones = [int(EPOCHS * 0.5), int(EPOCHS * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    train(model, optimizer, loss_fn, train_loader, valid_loader, EPOCHS, 
          metric_func = acc_fun,device = device,model_name = model_name,lr_scheduler = scheduler)