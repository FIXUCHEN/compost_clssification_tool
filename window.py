import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from models import ResNet18 , VGG16 , VGG19
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

model_name = 'resnet18' # resnet18 , vgg16 , vgg19 , choose 模型选择
if model_name == 'resnet18':
    model = ResNet18(16)
    model.load_state_dict(torch.load('saved_models/resnet18.pth',map_location=torch.device('cpu')))
elif model_name == 'vgg16':
    model = VGG16(16)
    model.load_state_dict(torch.load('saved_models/vgg16.pth',map_location=torch.device('cpu')))
elif model_name == 'vgg19':
    model = VGG19(16)
    model.load_state_dict(torch.load('saved_models/vgg19.pth',map_location=torch.device('cpu')))

class ImageClassifierApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        
        
        screen_width = self.master.winfo_screenwidth() // 2
        screen_height = self.master.winfo_screenheight() // 2
        
        self.master.geometry("%dx%d+0+0" % (screen_width, screen_height))
        self.master.resizable(0, 0)
        self.master.title("Compostable Classifier")
        # self.pack()
        
        self.max_width = screen_width
        self.max_height = screen_height
        self.create_widgets()
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # self.model.load_state_dict(torch.load('saved_models/resnet18.pth',map_location=self.device))
        self.model.eval()
        self.category = ['Carambola', 'Pitaya', 'apple', 'banana', 'cardboard', 'compost', 'glass', 'kiwi', 'mango', 'metal', 'orange', 'paper', 'peach', 'plastic', 'tamotoes', 'trash']

        self.super_category = {'pos':['Carambola', 'Pitaya', 'apple', 'banana', 'kiwi', 'mango', 'orange', 'peach', 'tamotoes', 'compost'],
                        'neg':['glass','metal','plastic','cardboard','paper','trash']
                        }
        self.super_category['pos'] = [self.category.index(i) for i in self.super_category['pos']]
        self.super_category['neg'] = [self.category.index(i) for i in self.super_category['neg']]

        

    def create_widgets(self):
        
        # black area initially
        x = int(self.max_width * 0.01)
        y = int(self.max_height * 0.01)
        w = int(self.max_width * 0.8)
        h = int(self.max_height * 0.8)
        
        init_image = ImageTk.PhotoImage(Image.new("RGB", (w, h), (0, 0, 0)))
        self.image_label = tk.Label(self.master, image=init_image)
        # set image_label in the left side of the window
        self.image_label.place(x=x, y=y, width=w, height=h)
        # self.image_label.pack()
        
        self.upload_button = tk.Button(self.master, text="Upload", command=self.choose_file)
        # set upload_button in the right side of the image_label
        x = int(self.max_width * 0.8 + self.max_width * 0.01)
        y = int(self.max_height * 0.1 + self.max_height * 0.01)
        # set background color of upload_button to white
        self.upload_button.place(x=x, y=y)
        # self.upload_button.pack()
        
        self.classify_button = tk.Button(self.master, text="Predict", command=self.classify_image)
        y = int(y + self.max_height * 0.1)
        self.classify_button.place(x=x, y=y)
        # self.classify_button.pack()
        

        
        self.result_label = tk.Label(self.master, text="Result:")
        y = int(y + self.max_height * 0.2)
        self.result_label.place(x=x, y=y)
        # self.result_label.pack()
       
        self.confidence_label = tk.Label(self.master, text="Confidence:")
        # self.confidence_label.pack()
        y = int(y + self.max_height * 0.1)
        
        self.confidence_label.place(x=x, y=y)

        self.super_label = tk.Label(self.master, text="Compostable:")
        y = int(y + self.max_height * 0.1)
        self.super_label.place(x=x, y=y)
        # self.confidence_label.pack()
    def choose_file(self):
        
        file_path = filedialog.askopenfilename()
        width, height = int(self.max_width * 0.85), int(self.max_height * 0.85)
        if file_path:
            
            image = Image.open(file_path)
            image = image.resize((width, height))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        
            self.now_img = image
        self.result_label.configure(text="Result:")
        self.confidence_label.configure(text="Confidence:")
        self.super_label.configure(text="Compostable:")

    def classify_image(self):
        # TODO: classification logic and add confidential index 在这里实现图片分类逻辑，输出结果和置信度到label中
    
        img = self.now_img
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.model(img)
            score = torch.max(torch.softmax(outputs, dim=1))
            preds = torch.argmax(outputs, dim=1)
        
        self.result_label.configure(text="Result: " + self.category[preds.item()])
        self.confidence_label.configure(text="Confidence: " + str(score.item()*100)[:5] + '%')
        if preds.item() in self.super_category['pos']:
            self.super_label.configure(text="Compostable: Yes")
        else:
            self.super_label.configure(text="Compostable: No")


    

root = tk.Tk()
app = ImageClassifierApp(master=root)
app.mainloop()
