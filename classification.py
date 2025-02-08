import streamlit as st 
import json
import numpy
from models.model import MyResNet
# from models.preprocessing import preprocessing_func
import torch
from PIL import Image
import time
import torchvision.transforms as T

with open("models/label_map.json", "r") as f:
    class_labels = json.load(f)

class_labels = {v: k for k, v in class_labels.items()}

def load_model():
    model = MyResNet()
    state_dict = torch.load('models/weights_model_resnet50_1.pt', map_location=torch.device('cpu'))
    
    # Убираем ключи, которые не соответствуют модели MyResNet
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Обновляем state_dict модели
    model_state_dict.update(pretrained_dict)

    # Загружаем обновлённый state_dict в модель
    model.load_state_dict(model_state_dict)

    model.eval()
    return model

model = load_model()


preprocessing_func = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.CenterCrop((64, 64))
])

def preprocessing(img):
    return preprocessing_func(img)

def predict(img):
    img = preprocessing(img)
    img = img.unsqueeze(0)
    start_time = time.time()  
    with torch.no_grad():
        pred = model(img)
        probabilities = torch.nn.functional.softmax(pred, dim=1)[0]
        top5_probs, top5_classes = torch.topk(probabilities, 5)
    elapsed_time = time.time() - start_time
    top5_labels = [class_labels[idx.item()] for idx in top5_classes]
    return top5_probs, top5_labels, elapsed_time

# Интерфейс Streamlit
st.title("Классификация изображений")
uploader = st.file_uploader("Загрузите изображение", type="jpg")
if uploader is not None:
    image = Image.open(uploader).convert("RGB")
    st.image(image, caption='Ваше изображение')
    top5_probs, top5_classes, elapsed_time = predict(image)

    st.write("Топ-5 предсказанных классов:")
    for i in range(5):
        st.write(f"Класс: {top5_classes[i]} - вероятность: {top5_probs[i].item() * 100:.2f}%")

    st.write(f"⏱ Время предсказания: {elapsed_time:.4f} секунд")