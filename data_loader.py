import os
import requests
import h5py
import torch
import torch.nn.functional as F

class GISTDataLoader:
    def __init__(self, filename="gist-960-euclidean.hdf5"):
        self.filename = filename
        self.url = "http://ann-benchmarks.com/gist-960-euclidean.hdf5"

    def _download(self):
        if not os.path.exists(self.filename):
            print(f"[Info] Downloading {self.filename}...")
            try:
                r = requests.get(self.url, stream=True)
                with open(self.filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"[Error] Download failed: {e}")
                raise

    def load_data(self, device='cpu', train_limit=50000, test_limit=1000):
        self._download()
        print(f"[Info] Loading data from {self.filename}...")
        with h5py.File(self.filename, 'r') as f:
            # 读取并归一化
            db = F.normalize(torch.from_numpy(f['train'][:train_limit]).float().to(device), p=2, dim=1)
            qs = F.normalize(torch.from_numpy(f['test'][:test_limit]).float().to(device), p=2, dim=1)
            print(f"[Info] Database: {db.shape}, Queries: {qs.shape}")
            return db, qs