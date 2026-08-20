import os.path
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


def pad_collate_bags(batch: List):
    """
    Pads variable-length bags into a batch of (B, N, D).
    """

    embeddings = [item['embedding'] for item in batch]
    labels = [item['label'] for item in batch]
    coords = [item['coords'] for item in batch]
    slide_id = [item['slide_id'] for item in batch]

    # Pad to largest tensor in batch
    # max_len = max(embedding.shape[0] for embedding in embeddings)
    # padded_embeddings = [torch.nn.functional.pad(t, pad=(0, 0, 0, max_len - t.shape[0])) for t in embeddings]
    # padded_embeddings = torch.stack(padded_embeddings)

    # Using nested tensors
    padded_embeddings = torch.nested.nested_tensor(embeddings, layout=torch.jagged)

    if labels[0] is not None:
        labels = torch.tensor(labels)

    return {
        "embedding": padded_embeddings,
        "label": labels,
        "coords": coords,
        "slide_id": slide_id
    }


class BiopsyDataset(Dataset):
    def __init__(self, metadata, feature_path, slide_path=None, sample: int = 4800, filename_template: str = None):
        self.metadata = metadata
        self.labels = []
        self.features = []
        self.coordinates = []
        self.slide_ids = []
        self.feature_path = feature_path
        self.slide_path = slide_path
        self.sample = sample
        # Supplied via configuration (not hardcoded) since the institution-specific
        # slide filename format is privacy sensitive. Must contain `{case}` and
        # `{specimen}` placeholders, e.g. "{case}T{specimen}-".
        self.filename_template = filename_template

        feature_files = list(Path(self.feature_path).glob("*.h5"))
        feature_files = [f.name for f in feature_files]

        # If csv file is provided filter out features that are not in the csv file
        if self.metadata is not None:
            # metadata = self._load_metadata()
            feature_files = pd.Series(feature_files)
            for _, f in self.metadata.iterrows():
                prefix = self._build_prefix(f)
                matches = feature_files[feature_files.str.startswith(prefix)]

                if matches.any():
                    assert len(matches) == 1, f"More than one slide for biopsy {prefix}"
                    self.features.append(matches.iloc[0])
                    label = self._get_label(f)
                    self.labels.append(label)

            self._convert_labels()
            assert len(self.labels) == len(self.features)
        else:
            self.features = np.array(feature_files)
            print(self.features)

    def _build_prefix(self, row) -> str:
        """
        Builds the embedding-filename prefix for a metadata row from a template
        supplied via configuration, so the institution-specific slide filename
        format is never hardcoded here.
        """
        case = str(row.Report).replace("_", "")
        template = self.filename_template or "{case}{specimen}"
        return template.format(case=case, specimen=row.Specimen)

    def _load_metadata(self):
        metadata = pd.read_csv(self.csv_path, sep=";",
                              dtype={"GleasonPrimary": "Int32", "GleasonSecondary": "Int32"})
        return metadata

    def _get_label(self, row):
        raise NotImplementedError

    def _convert_labels(self):
        raise NotImplementedError

    def _load_embeddings(self, index):
        p = os.path.join(self.feature_path, self.features[index])
        f = h5py.File(p, "r")
        embedding = f["features"][:]
        coords = f["coords"][:]
        assert embedding.ndim == 2
        embedding = torch.from_numpy(embedding)
        coords = torch.from_numpy(coords)
        return embedding, coords

    def __getitem__(self, index):
        embedding, coords = self._load_embeddings(index)
        emb_len = len(embedding)

        if 0 < self.sample < emb_len:
            permutation = np.random.permutation(embedding.shape[0])[:self.sample]
            embedding = embedding[permutation]
            coords = coords[permutation]

        if len(self.labels) > 0:
            label = self.labels[index]
        else:
            label = None

        return {
            "embedding": embedding,
            "label": label,
            "coords": coords,
            "slide_id": self.features[index]
        }

    def __len__(self):
        return len(self.features)


class BiopsyGleasonDataset(BiopsyDataset):
    def __init__(self, metadata, feature_path, slide_path, filename_template: str = None):
        self.cls_to_idx = {
            "benign": 0,
            "6": 1,
            "7a": 2,
            "7b": 3,
            "8": 4,
            "9": 5,
        }
        self.idx_to_cls = {cls: label for label, cls in self.cls_to_idx.items()}
        super().__init__(metadata, feature_path, slide_path, filename_template=filename_template)

    def _convert_labels(self):
        self.labels = [self.cls_to_idx[label] for label in self.labels]
        self.labels = np.array(self.labels)

    def _get_label(self, row):
        return str(row.GleasonScore)


class BiopsyPandasDataset:
    def __init__(self, metadata, feature_path, slide_path):
        pass
