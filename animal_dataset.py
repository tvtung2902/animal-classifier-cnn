from torch.utils.data import Dataset
import os
from PIL import Image

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        if train:
            mode = 'train'
        else:
            mode = 'test'
        self.root = os.path.join(root, mode)
        self.categories = os.listdir(self.root)
        self.image_paths = []
        self.labels = []
        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)
            print(data_file_path)
            for filename in os.listdir(data_file_path):
                self.image_paths.append(os.path.join(data_file_path, filename))
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    @staticmethod
    def get_categories():
        return ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
if __name__ == '__main__':
    root = '../all-data/animals'
    dataset = AnimalDataset(root, train=True)
    print(dataset.__len__())
    image, label = dataset.__getitem__(0)
    print(image.show())
    print(label)