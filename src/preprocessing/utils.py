import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import librosa as lb

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift
)

GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def create_dataset_splits(root_folder, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, files_per_class=100):
    """
    Create train, validation, and test splits for audio files while maintaining class balance
    and keeping files ordered by class.
    
    Args:
        root_folder (str): Path to the root folder containing class subfolders
        train_ratio (float): Ratio of files to use for training (default: 0.7)
        val_ratio (float): Ratio of files to use for validation (default: 0.1)
        test_ratio (float): Ratio of files to use for testing (default: 0.2)
        files_per_class (int): Number of files in each class (default: 100)
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Calculate number of files for each split
    n_train = int(files_per_class * train_ratio)  # 70 files
    n_val = int(files_per_class * val_ratio)      # 10 files
    n_test = files_per_class - n_train - n_val    # 20 files
    
    # Get sorted list of subfolders (classes)
    class_folders = sorted([d for d in os.listdir(root_folder) 
                          if os.path.isdir(os.path.join(root_folder, d))])
    
    # Initialize lists for each split
    train_files = []
    val_files = []
    test_files = []
    
    # Process each class in order
    for class_name in class_folders:
        class_path = os.path.join(root_folder, class_name)
        
        # Get sorted list of audio files
        audio_files = sorted([f for f in os.listdir(class_path)
                            if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))])
        
        assert len(audio_files) == files_per_class, f"Expected {files_per_class} files in {class_name}, found {len(audio_files)}"
        
        # Create relative paths
        class_files = [os.path.join(class_name, f) for f in audio_files]
        
        # Shuffle files while maintaining reproducibility
        random.seed(hash(class_name))  # Use class name as seed for consistent shuffling
        random.shuffle(class_files)
        
        # Split files
        train_files.extend(class_files[:n_train])
        val_files.extend(class_files[n_train:n_train + n_val])
        test_files.extend(class_files[n_train + n_val:])
    
    # Write splits to files, maintaining class order
    def write_split(file_name, files):
        with open(file_name, 'w', encoding='utf-8') as f:
            for file_path in files:
                f.write(f"{file_path}\n")
    
    write_split('../../datasets/train.txt', train_files)
    write_split('../../datasets/val.txt', val_files)
    write_split('../../datasets/test.txt', test_files)
    
    # Print statistics
    print(f"Dataset split complete:")
    print(f"Train set: {len(train_files)} files ({n_train} per class)")
    print(f"Validation set: {len(val_files)} files ({n_val} per class)")
    print(f"Test set: {len(test_files)} files ({n_test} per class)")
    
    # Print class distribution in order
    print("\nClass distribution (in order):")
    for class_name in class_folders:
        train_count = sum(1 for f in train_files if class_name in f)
        val_count = sum(1 for f in val_files if class_name in f)
        test_count = sum(1 for f in test_files if class_name in f)
        print(f"{class_name}:")
        print(f"  Train: {train_count}")
        print(f"  Val: {val_count}")
        print(f"  Test: {test_count}")


class GTZANDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path =  data_path if data_path else ''
        self.split = split
        # self.is_augmentation = is_augmentation
        self.genres = GTZAN_GENRES
        self._get_song_list()
        # if is_augmentation:
        #     self._get_augmentations()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, '%s.txt' % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5)
        ])
        self.augmentation = augment(self.samples, self.sample_rate)

    # def _adjust_audio_length(self, wav):
    #     if self.split == 'train':
    #         random_index = random.randint(0, len(wav) - self.num_samples - 1)
    #         wav = wav[random_index : random_index + self.num_samples]
    #     else:
    #         hop = (len(wav) - self.num_samples) // self.num_chunks
    #         wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)])
    #     return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get genre
        genre_name = line.split('/')[0]
        genre_index = self.genres.index(genre_name)

        # get audio
        audio_filename = os.path.join(self.data_path, 'genres', line)
        self.samples, self.sample_rate = lb.load(audio_filename)


        # # adjust audio length
        # wav = self._adjust_audio_length(wav).astype('float32')

        # data augmentation
        # if self.is_augmentation:
        #     samples = self.augmentation(torch.from_numpy(self.samples).unsqueeze(0)).squeeze(0).numpy()

        return genre_index

    def __len__(self):
        return len(self.song_list)

def get_dataloader(data_path=None, 
                   split='train', 
                   batch_size=16, 
                   num_workers=0
                   ):
    is_shuffle = True if (split == 'train') else False
    data_loader = DataLoader(dataset=GTZANDataset(data_path, 
                                                       split 
                                                       ),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader