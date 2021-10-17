import argparse
import pathlib
from tqdm import tqdm

import numpy as np
import soundfile as sf


def make_dataset(base_path, fnames, save_path):
    for fname in tqdm(fnames):
        noisy, _ = sf.read(base_path.joinpath('noisy_trainset_wav', fname))
        clean, _ = sf.read(base_path.joinpath('clean_trainset_wav', fname))

        npy_name = pathlib.Path(save_path).joinpath(fname).with_suffix('.npy')
        with open(npy_name, "wb") as f:
            np.save(f, noisy.astype(np.float32))
            np.save(f, clean.astype(np.float32))


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('base_data_dir', type=str)
    parser.add_argument('base_save_dir', type=str)
    parser.add_argument('--num_train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    # Prepare file names
    base_path = pathlib.Path(args.base_data_dir)
    clean_path = base_path.joinpath('clean_trainset_wav')
    fnames = np.sort([fname.parts[-1] for fname in  clean_path.glob('*.wav')])
    fnames = np.random.permutation(fnames)

    # Split dataset
    tmp = int(len(fnames)*args.num_train_ratio)
    train_fnames = fnames[:tmp]
    valid_fnames = fnames[tmp:]

    # Run make_dataset
    make_dataset(base_path, fnames=train_fnames, save_path=args.base_save_dir+'/train')
    make_dataset(base_path, fnames=valid_fnames, save_path=args.base_save_dir+'/valid')


if __name__ == '__main__':
    main()