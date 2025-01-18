import os
import random
import numpy as np
import librosa as lb
from torch.utils.data import DataLoader, Dataset
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift
)

# from torchaudio_augmentations import (
#     RandomResizedCrop,
#     RandomApply,
#     PolarityInversion,
#     Noise,
#     Gain,
#     HighLowPass,
#     Delay,
#     PitchShift,
#     Reverb,
#     Compose,
# )


class GTZANDataset(Dataset):
    def __init__(self, data_path, split, num_samples, num_chunks):
        self.data_path =  data_path if data_path else ''
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        # self.is_augmentation = is_augmentation
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

    def _adjust_audio_length(self, wav):
        if self.split == 'train':
            random_index = random.randint(0, len(wav) - self.num_samples - 1)
            wav = wav[random_index : random_index + self.num_samples]
        else:
            hop = (len(wav) - self.num_samples) // self.num_chunks
            wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)])
        return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get genre
        genre_name = line.split('\\')[0]
        genre_index = self.genres.index(genre_name)

        # get audio
        audio_filename = os.path.join(self.data_path, "genres_original", line)
        wav, sample_rate = lb.load(audio_filename)


        # adjust audio length
        wav = self._adjust_audio_length(wav).astype('float32')

        # data augmentation
        # if self.is_augmentation:
        #     samples = self.augmentation(torch.from_numpy(self.samples).unsqueeze(0)).squeeze(0).numpy()

        return wav, genre_index

    def __len__(self):
        return len(self.song_list)

def get_dataloader(data_path=None, 
                   split='train',
                   num_samples=22050 * 29, 
                   num_chunks=1,
                   batch_size=16, 
                   num_workers=0
                   ):
    is_shuffle = True if (split == 'train') else False
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)
    data_loader = DataLoader(dataset=GTZANDataset(data_path, 
                                                    split,
                                                    num_samples, 
                                                    num_chunks
                                                ),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader