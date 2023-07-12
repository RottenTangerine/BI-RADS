import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    features = []
    for child in root.find('tumor'):
        if child.tag in ['shape', 'margin', 'shadow']:
            features.append(child.text)
        else:
            features.append(float(child.text))

    isBenign = root.find('isBenign').text
    if isBenign == "False":
        label = 0
    else:
        label = 1

    return features, label


class TumorDataset(Dataset):
    def __init__(self, img_dir, xml_dir, transform=None):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

        self.feature_dict = {
            "shape": {
                "ellipse": 0,
                # add more shapes here if exist
            },
            "margin": {
                "Microlobulated": 0,
                # add more margin types here if exist
            },
            "shadow": {
                "black": 0,
                # add more shadow types here if exist
            }
        }

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace('.jpg', '.xml'))

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        features, label = parse_xml(xml_path)

        # convert features to vector
        feature_vector = []
        feature_names = ['shape', 'margin', 'center_x', 'center_y', 'radius_a', 'radius_b',
                         'angel', 'density', 'edge_r', 'edge_num', 'shadow', 'spot_num']
        for i, feature in enumerate(features):
            if feature_names[i] in self.feature_dict:
                feature_vector.append(self.feature_dict[feature_names[i]][feature])
            else:
                feature_vector.append(feature)

        feature_vector = torch.tensor(feature_vector)

        return image, feature_vector, label


from torch.utils.data import DataLoader

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config import load_config

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = TumorDataset("./ellipse_malignant", "./ellipse_malignant_annotation", transform=transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    data = next(iter(dataloader))
    img, feature_vector, label = data

    print("Feature Vector: ", feature_vector)
    print("Label: ", label)

    # 显示一张图像
    img = img[0].permute(1, 2, 0)
    plt.imshow(img)
    plt.show()