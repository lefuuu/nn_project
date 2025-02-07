import streamlit as st 
import numpy
from models.model import MyResNet
# from models.preprocessing import preprocessing_func
import torch
from PIL import Image
import time
import torchvision.transforms as T

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
    with torch.no_grad():
        pred = model(img)
    return pred

# Интерфейс Streamlit
st.title("Классификация изображений")
uploader = st.file_uploader("Загрузите изображение", type="jpg")
if uploader is not None:
    image = Image.open(uploader).convert("RGB")
    st.image(image, caption='Ваше изображение')

    prediction = predict(image)
    st.write(prediction)