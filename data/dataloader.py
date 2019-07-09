import os
import numpy as np
import torch
import scipy.signal
import librosa
import fnmatch

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from multiprocessing.pool import Pool as ThreadPool
from tqdm import tqdm

from global_constants import *
from custom_logging import get_logger


dlogger = get_logger(__name__)


windows = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class MusicDataset(Dataset):
    def __init__(self, opt, mode, type=MFCC, debug=False, quiet=False, pre_cache=False):
        super(MusicDataset, self).__init__()
        self.mode = mode  # to load train/val/test data
        self.type = type
        self.quiet = quiet
        # load the json file which contains information about the dataset
        self.dataset_manifest = []
        self.dataset_cache = {}
        for mp3_root in (opt["root_prog"], opt["root_nonprog"]):
            for root, dirs, filenames in os.walk(mp3_root):
                for ffn in fnmatch.filter(filenames, '*.mp3'):
                    full_path = os.path.join(root, ffn)
                    if mp3_root == opt["root_prog"]:
                        target = 1.0
                    else:
                        target = 0.0

                    self.dataset_manifest.append((full_path, target))

        if debug:
            np.random.shuffle(self.dataset_manifest)
            self.dataset_manifest = self.dataset_manifest[:32]

        self.n = len(self.dataset_manifest)

        # Some params borrowed from DeepSpeech project : https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        # Sample rate 44100 for music (voice is 16000)
        self.disk_sample_rate = 44100
        self.sample_rate = self.disk_sample_rate
        self.window_size = 0.02  # Window size for spectrogram in seconds
        self.window_stride = 0.01  # Window stride for spectrogram in seconds
        self.n_fft = int(self.sample_rate * self.window_size)
        self.window = windows['hamming']  # Window type for spectrogram generation
        self.normalize = True

        num_train = int(train_split * self.n)
        num_val = int(val_split * self.n)
        self.splits = {  # TODO(WG)
            'train': list(np.arange(0, num_train)),
            'val': list(np.arange(num_train, num_train + num_val)),
            'test': list(np.arange(num_train + num_val, self.n))
        }

        if not quiet:
            dlogger.debug('number of train songs:\t{}'.format(len(self.splits['train'])))
            dlogger.debug('number of val songs:\t{}'.format(len(self.splits['val'])))
            dlogger.debug('number of test songs:\t{}'.format(len(self.splits['test'])))
            # self.n_frame_steps = opt['n_frame_steps']

        if pre_cache:
            dlogger.info("Pre-caching STFT features...")
            jobs = []
            for ix in self.splits[self.mode]:
                full_path, target = self.dataset_manifest[ix]
                jobs.append((ix, full_path))

            pool = ThreadPool(opt["num_workers"])
            res = pool.imap_unordered(self.cache_feat_job, jobs)
            for ix, spect in tqdm(res):
                self.dataset_cache[ix] = spect

        dlogger.info("Finished initializing dataloader.")

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = ix % len(self.splits[self.mode])
        full_path, target = self.dataset_manifest[ix]
        if not self.quiet: dlogger.debug('Load ix={}'.format(ix))
        # sample_id = 'sample_{}'.format(self.splits[self.mode][ix])
        # sample_query = self.h5file.root[sample_id]

        data = {}
        if self.dataset_cache.get(ix) is not None:
            data['x'] = self.dataset_cache[ix]
        elif self.type == 'log':
            # x = sample_query.raw[0]
            x = self.load_audio(full_path)
            spect = self.extract_feature(x)

            data['x'] = spect
            self.dataset_cache[ix] = spect
        else:
            # Caching raw array won't be viable
            x = self.load_audio(full_path)
            data['x'] = Variable(torch.from_numpy(x).type(torch.FloatTensor))

        data['gt'] = Variable(torch.tensor(float(target)))
        data['sample_id'] = ix

        return data

    def __len__(self):
        return len(self.splits[self.mode])

    def load_audio(self, full_path):
        x, _ = librosa.load(full_path, sr=self.sample_rate, mono=True, offset=10.0, duration=120.0)
        return x

    def extract_feature(self, x):
        num_sec = float(len(x)) / self.disk_sample_rate
        num_min = int(np.ceil(float(num_sec) / 60.))
        # Taking the whole music sample runs out of memory (10 minute ~= 6GB RAM).
        # Instead, concat some small 10 second windows. Take 10 seconds @ 16khz from every minute of audio
        xp = np.zeros((disk_sample_rate * num_min * 60))
        xp[:len(x)] = x
        xp = xp.reshape(num_min, disk_sample_rate * 60)
        # sub-sample minute-wise. 10 seconds per minute, from end to avoid intro
        xp = xp[:, -disk_sample_rate * 10:]
        xp = np.concatenate(xp, axis=0)
        x = xp

        # Short-time Fourier transform (STFT) spectrogram (complex valued)
        window_length = self.n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        n_fft = self.n_fft

        D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length,
                         win_length=window_length, window=self.window)

        # Separate a complex-valued spectrogram D into its magnitude (S) and phase (P) components, so that D = S * P.
        spect, phase = librosa.magphase(D)

        # S = log(S + 1)
        # Return the natural logarithm of one plus the input array, element-wise.
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def cache_feat_job(self, meta):
        ix, fpath = meta
        if not self.quiet: dlogger.debug('Load ix={}'.format(ix))
        x = self.load_audio(fpath)
        spect = self.extract_feature(x)
        return ix, spect


def _collate_fn(batch):
    """
    Modified version of DS2 collate
    :param batch: The mini batched data from dataset __getitem__
    :return:
    """

    def func(p):
        return p['x'].size(1)

    batch = sorted(batch, key=lambda sample: sample['x'].size(1), reverse=True)
    longest_sample = max(batch, key=func)['x']
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    # Pseudo-image shape - BxCxDxT: Batch x Channel (1 channel) x Freq bins x Time bins
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample['x']
        target = sample['gt']
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        targets.append(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages


class MusicDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Data loader for MusicDataset to replace collate_fn
        """
        super(MusicDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
