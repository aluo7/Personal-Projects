# data.py

import os
import requests
import tarfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from torchvision.datasets.video_utils import VideoClips
import glob
import zipfile

def download_and_extract_imagenette(data_dir='../data'):
    os.makedirs(data_dir, exist_ok=True)
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    dataset_path = os.path.join(data_dir, 'imagenette2-320.tgz')

    if not os.path.exists(os.path.join(data_dir, 'imagenette2-320')):
        response = requests.get(dataset_url, stream=True)
        with open(dataset_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Imagenette download complete.")
    else:
        print("Imagenette already exists, skipping download.")

def get_imagenette_dataloader(batch_size=32, img_size=224, data_dir='../data'):
    download_and_extract_imagenette(data_dir=data_dir)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'imagenette2-320/val'),
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def download_and_extract_ucf101(data_dir='../data', max_videos=10):
    ucf101_dir = os.path.join(data_dir, 'UCF101')
    os.makedirs(ucf101_dir, exist_ok=True)

    categories = {
        'ApplyEyeMakeup': [
            'v_ApplyEyeMakeup_g01_c01.avi', 
            'v_ApplyEyeMakeup_g01_c02.avi'
        ],
        'BoxingPunchingBag': [
            'v_BoxingPunchingBag_g01_c01.avi',
            'v_BoxingPunchingBag_g01_c02.avi'
        ]
    }
    
    for category, videos in categories.items():
        category_dir = os.path.join(ucf101_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for video_name in videos[:max_videos]:
            video_url = f"https://crcv.ucf.edu/THUMOS14/UCF101/UCF101/{category}/{video_name}"
            video_path = os.path.join(category_dir, video_name)
            
            if not os.path.exists(video_path):
                response = requests.get(video_url, stream=True, verify=False)
                
                if response.status_code == 200:
                    with open(video_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f"download complete: {video_name}")
                else:
                    print(f"failed to download {video_name} from {video_url}")
            else:
                print(f"{video_name} already exists, skipping download.")

def download_ucf101_annotations(data_dir='../data'):
    annotation_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    os.makedirs(annotation_dir, exist_ok=True)

    annotation_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    annotation_zip_path = os.path.join(annotation_dir, 'UCF101TrainTestSplits-RecognitionTask.zip')

    if not os.path.exists(annotation_zip_path):
        response = requests.get(annotation_url, stream=True, verify=False)
        with open(annotation_zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print("annotations download complete.")

    if not os.path.exists(os.path.join(annotation_dir, 'trainlist01.txt')):
        with zipfile.ZipFile(annotation_zip_path, 'r') as zip_ref:
            zip_ref.extractall(annotation_dir)
        print("annotations extracted")

def get_ucf101_dataloader(batch_size=4, img_size=112, data_dir='../data', num_frames=16):
    def custom_transform(video):
        if video.dim() == 4 and video.size(1) == 3:
            pass
        else:
            video = video.permute(0, 3, 1, 2)

        resized_frames = torch.stack([transforms.Resize((img_size, img_size))(frame) for frame in video])
        resized_frames = resized_frames.to(torch.float32) / 255.0
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # norm
        normalized_video = torch.stack([normalize(frame) for frame in resized_frames])
        
        return normalized_video

    transform = transforms.Compose([
        transforms.Lambda(custom_transform)  # apply custom transformation
    ])

    class UCF101Dataset(UCF101):
        def __init__(self, root, annotation_path, frames_per_clip=num_frames, transform=None):
            video_paths = []  # manually init video_clips due for double call override
            for root_dir, _, files in os.walk(root):
                video_paths.extend([os.path.join(root_dir, f) for f in files if f.endswith('.avi')])
            
            self.manual_video_clips = VideoClips(
                video_paths,
                clip_length_in_frames=frames_per_clip,
                frames_between_clips=1
            )

            super().__init__(root, annotation_path, frames_per_clip=frames_per_clip, transform=transform)

            if hasattr(self, 'video_clips') and len(self.video_clips.video_paths) == 0:
                print("reassigning manually initialized videoclips")
                self.video_clips = self.manual_video_clips

            if hasattr(self, 'samples') and hasattr(self, 'video_clips'):
                print("reassigning samples and indices based on the manually initialized videoclips")
                self.samples = [(path, self.manual_video_clips.video_paths.index(path)) for path in self.manual_video_clips.video_paths]
                self.indices = list(range(len(self.samples)))

            self.transform = transform

        def __getitem__(self, idx):
            video, _, label = super().__getitem__(idx)
            return video, label

    download_and_extract_ucf101(data_dir=data_dir)
    download_ucf101_annotations(data_dir=data_dir)

    annotation_path = os.path.join(data_dir, 'ucfTrainTestlist')
    
    dataset = UCF101Dataset(
        root=os.path.join(data_dir, 'UCF101'),
        annotation_path=annotation_path,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
