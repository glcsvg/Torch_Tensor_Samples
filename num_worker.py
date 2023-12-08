import numpy as np
from p_detect import Custom_Reid_Data
from torchvision import transforms
import multiprocessing as mp
import torch
from time import time

class DataLoaderNumWorker:
    def __init__(self):
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_datasets = {}

    def create_img(self):
        random_image = []
        labels = []
        for i in range(0, 20):
            rgb = np.random.randint(255, size=(256, 128, 3), dtype=np.uint8)
            random_image.append([rgb])
            labels.append(0)
        return random_image, labels

    def prepare_data(self):
        data = {x: self.create_img() for x in ['gallery', 'query']}
        self.image_datasets['query'] = Custom_Reid_Data(data['query'][0], self.data_transforms)
        self.image_datasets['gallery'] = Custom_Reid_Data(data['gallery'][0], self.data_transforms)

    def best_worker(self):
        worker_list = []
        for num_workers in range(0, mp.cpu_count(), 2):
            # Dataloader
            dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4,
                                                          shuffle=False, num_workers=num_workers) for x in
                           ['gallery', 'query']}
            start = time()
            for epoch in range(1, 2):
                for i, data in enumerate(dataloaders, 0):
                    pass
            end = time()
            elep_time = end - start
            worker_list.append(elep_time)

        worker = np.argmin(worker_list)
        return worker * 2


# Usage:
loader = DataLoaderNumWorker()
loader.prepare_data()
num = loader.best_worker()
print(f"Num worker:{num}")
