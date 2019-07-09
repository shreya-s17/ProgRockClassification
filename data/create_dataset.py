import os
import numpy as np
import tables
import argparse
import fnmatch
import librosa

from tqdm import tqdm
from data.feature_extraction import getMFCCS, normalizeInputs
from data.infrastructure import *
from custom_logging import get_logger

cd_logger = get_logger(__name__)


def create_dataset(opt):
    # flac_root = opt.flac_root
    prog_mp3_root = opt.prog_mp3_root
    nonprog_mp3_root = opt.nonprog_mp3_root
    # if flac_root is None:
    #     cd_logger.info("Not sourcing metadata for this dataset")

    cd_logger.info("Sourcing audio from:")
    cd_logger.info("Non-prog:\t{}".format(nonprog_mp3_root))
    cd_logger.info("Prog:\t{}".format(prog_mp3_root))

    out_filepath = os.path.join(opt.out_dir, 'dataset_bank.h5')

    with tables.open_file(out_filepath, mode='w', title="MP3 Audio Dataset") as h5file:
        h5root = h5file.root
        cd_logger.info(h5file)

        i = 0

        for mp3_root in (nonprog_mp3_root, prog_mp3_root):
            paths = []
            for root, dirs, filenames in os.walk(mp3_root):
                for ffn in fnmatch.filter(filenames, '*.mp3'):
                    full_path = os.path.join(root, ffn)
                    paths.append(full_path)

            for full_path in tqdm(paths): #[:20]:
                x, sr = librosa.load(full_path, sr=None, mono=True)
                _, _, mfccs = getMFCCS(x, sr, 20)

                x = np.expand_dims(x, 0)
                mfccs = np.expand_dims(mfccs, 0)

                # file compression, use zlib default
                filters = tables.Filters(complevel=5)
                # h5file.root.sample_i
                sample_group = h5file.create_group(h5root, 'sample_{}'.format(i), full_path, filters=filters)

                # h5file.root.sample_i.raw
                music_array = h5file.create_earray(sample_group, 'raw', obj=x)
                # h5file.root.sample_i.mfccs
                mfccs_array = h5file.create_earray(sample_group, 'mfccs', obj=mfccs)

                music_array.append(x)
                mfccs_array.append(mfccs)

                sample_group._v_attrs.concept = int((mp3_root == prog_mp3_root))
                sample_group._v_attrs.number_id = i

                i += 1

        cd_logger.info("Done")


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--out_dir',
                     help='Output directory of dataset_bank.h5',
                     required=True)
    # opt.add_argument('--flac_root',
    #                  help='root of .flac files. Can be sourced for meta data.',
    #                  required=False,
    #                  default=None)
    opt.add_argument('--prog_mp3_root',
                     help='root of prog .mp3 files. Sourced for audio data.',
                     required=True)
    opt.add_argument('--nonprog_mp3_root',
                     help='root of non-prog .mp3 files. Sourced for audio data.',
                     required=True)

    opt = opt.parse_args()

    create_dataset(opt)

