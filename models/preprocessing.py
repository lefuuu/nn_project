import torchvision.transforms as T
import torch
import numpy as np

preprocessing_func = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.CenterCrop((64, 64))
])

# def preprocessing(img):
#     img = img.convert("RGB")  # Приведение в RGB
#     img = preprocessing_func(img)  # Применяем resize и crop
#     img = np.array(img, dtype=np.float32) / 255.0  # Преобразуем в numpy-массив
#     img = torch.tensor(img).permute(2, 0, 1)  # Меняем оси на (C, H, W)
#     return img