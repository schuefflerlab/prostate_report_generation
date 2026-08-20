from pathlib import Path
import re
from typing import Optional, Tuple, List
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from open_clip_train.data import DataInfo


class TextDataset(Dataset):
    def __init__(self, embeddings_paths: list[str], text_path: str = None, tokenizer = None, sample: int = 1000, split: str = 'train', split_file: str = None, oversample: bool = False, class_file: str = None, text_id_pattern: str = None, embedding_id_pattern: str = None):
        super().__init__()
        assert split in ("train", "val", "test"), "Invalid split"
        self.tokenizer = tokenizer
        self.sample = sample
        self.inference = text_path is None
        self.oversample = oversample
        # Regexes are supplied via configuration (not hardcoded) since the
        # institution-specific report/slide identifier format is privacy sensitive.
        # Each must define named groups `case` and `specimen`.
        self.text_id_pattern = text_id_pattern
        self.embedding_id_pattern = embedding_id_pattern

        embedding_files = []
        for ep in embeddings_paths:
            embedding_files.extend(list(Path(ep).glob("*.h5")))

        assert len(embedding_files) > 0, "No embedding files found"

        e_samples = {}
        for e in embedding_files:
            e_samples.setdefault(self._meta_from_embedding(e.stem), []).append(e)

        if split_file is not None:
            # Read split data
            df = pd.read_csv(split_file)
            self.data = set(df[df['split'] == split]["id"].tolist())

            e_samples = {sid: p for sid, p in e_samples.items() if p[0].stem in self.data}

        if not self.inference:
            self.text_path = Path(text_path)
            text_files = list(self.text_path.glob("*.txt"))
            t_samples = {self._meta_from_text(e.stem): e for e in text_files}

            # Only keep samples where embedding and text is available
            e_samples = {k: v for k, v in e_samples.items() if k in set(t_samples)}
            t_samples = {k: v for k, v in t_samples.items() if k in set(e_samples)}
            assert len(e_samples) == len(t_samples), "Number of embeddings and texts must be equal"
            t_samples = dict(sorted(t_samples.items()))
            e_samples = dict(sorted(e_samples.items()))

            if len(e_samples) > 0:
                assert list(e_samples.keys())[-1] == list(t_samples.keys())[-1], "Last key must be equal"
            self.text_files = [(k, v) for k, v in t_samples.items()]

        self.embedding_files = [(k, v) for k, v in e_samples.items()]

        if class_file is not None:
            self.weights, self.classes = self._get_weights(class_file, self.embedding_files)
        else:
            self.weights, self.classes = None, None

        if len(self.embedding_files) > 0:
            (first_case, first_specimen), _ = self.embedding_files[0]
            print(f"First sample: {first_case}_{first_specimen} (must be equal)")
        else:
            print("Empty dataset")


    def _meta_from_text(self, filename: str) -> Tuple[str, str]:
        return self._parse_id(filename, self.text_id_pattern)

    def _meta_from_embedding(self, filename: str) -> Tuple[str, str]:
        return self._parse_id(filename, self.embedding_id_pattern)

    @staticmethod
    def _parse_id(filename: str, pattern: Optional[str]) -> Tuple[str, str]:
        """
        Extracts a (case, specimen) identifier pair from a filename using a regex
        supplied via configuration. The pattern must define named groups `case` and
        `specimen`. If no pattern is configured, the filename is used as an opaque
        case id (no specimen splitting).
        """
        if not pattern:
            return filename, None

        m = re.search(pattern, filename)
        if m is None:
            return filename, None

        case = re.sub(r"[^A-Za-z0-9]", "", m.group("case"))
        specimen = m.group("specimen")
        return case, specimen

    @staticmethod
    def _oversample_patches(features, coords, n: int = 1):
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        x_extent = x_max - x_min
        base_shift = 2 * x_extent

        features_oversampled = np.tile(features, (n + 1, 1))

        shifts = np.arange(n + 1) * base_shift  # [0, base_shift, 2*base_shift, ...]
        coords_oversampled = np.vstack([coords + np.array([s, 0.0]) for s in shifts])
        return features_oversampled, coords_oversampled

    @staticmethod
    def _get_weights(class_file, samples):
        df = pd.read_csv(class_file, sep=";")
        report_ids = df["Report"].str.replace("_", "").tolist()
        specimen = df["Specimen"].astype(str).tolist()

        classes = df['GleasonScore'].tolist()
        targets = df['GleasonScore'].astype('category').cat.codes.tolist()
        samples_weights = {(r, s): (t, c) for r, s, t, c in zip(report_ids, specimen, targets, classes)}

        values, counts = torch.unique(torch.tensor(targets), return_counts=True)
        top_idx = torch.argmax(counts)
        majority_class = values[top_idx]

        targets_filtered = []
        samples_classes = []
        missed_weights = 0
        for s, _ in samples:
            try:
                target, class_name = samples_weights[s]
                targets_filtered.append(target)
                samples_classes.append(class_name)
            except KeyError:
                targets_filtered.append(majority_class)
                samples_classes.append(None)
                missed_weights += 1

        print(f"Missed weights: {missed_weights}")

        targets_filtered = torch.tensor(targets_filtered)

        class_sample_count = torch.tensor([(targets_filtered == t).sum() for t in torch.unique(targets_filtered, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets_filtered])

        assert len(samples_weight) == len(samples_classes)

        return samples_weight, samples_classes


    def __getitem__(self, index: int):
        (case_emb, specimen_emb), embedding_files = self.embedding_files[index]
        embedding_file = random.choice(embedding_files)
        if self.classes is not None:
            cls = self.classes[index]
        else:
            cls = None

        with h5py.File(embedding_file, "r") as embedding:
            features = embedding["features"][:]
            coords = embedding["coords"][:]

        report_id = case_emb + ("_" + specimen_emb if specimen_emb is not None else "")

        emb_len = len(features)

        if emb_len < 1500 and self.oversample:
            features, coords = self._oversample_patches(features, coords, n=6)
            emb_len = len(features)

        if 0 < self.sample < emb_len:
            permutation = np.random.permutation(features.shape[0])[:self.sample]
            features = features[permutation]
            coords = coords[permutation]

        coords_orig = torch.tensor(coords, dtype=torch.float32)

        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

        features = torch.tensor(features, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)

        cutoff = max(0, emb_len - self.sample)

        if not self.inference:
            (case_text, specimen_text), text_file = self.text_files[index]
            assert case_text == case_emb
            assert specimen_text == specimen_emb

            with open(text_file, "r") as f:
                text = f.read()

            text = self.tokenizer(text)[0]
        else:
            text = None

        return report_id, features, coords, coords_orig, text, cutoff, cls

    def __len__(self):
        return len(self.embedding_files)



def collate_batch_padded(batch: List):
    batch_labels = []
    batch_embeddings = []
    batch_coords = []
    batch_coords_orig = []
    batch_texts = []
    cuttoffs = []
    classes = []

    for label, features, coords, coords_orig, text, cutoff, cls in batch:
        batch_labels.append(label)
        batch_embeddings.append(features)
        batch_coords.append(coords)
        batch_coords_orig.append(coords_orig)
        batch_texts.append(text)
        cuttoffs.append(cutoff)
        classes.append(cls)

    max_emb_len = max([len(e) for e in batch_embeddings])
    # Set to a multiple of 64
    max_emb_len = ((max_emb_len - 1) // 64 + 1) * 64

    assert len(batch_embeddings[0]) == len(batch_coords[0]), "Embeddings and coords don't have the same length"

    padded_batch_embeddings = []
    for e in batch_embeddings:
        assert len(e.shape) == 2
        e_padded = F.pad(e, (0, 0, 0, max_emb_len - e.shape[0]))
        padded_batch_embeddings.append(e_padded)

    padded_coords = []
    for c in batch_coords:
        assert len(c.shape) == 2
        c_padded = F.pad(c, (0, 0, 0, max_emb_len - c.shape[0]))
        padded_coords.append(c_padded)

    embeddings = torch.stack(padded_batch_embeddings, dim=0)

    # If inference, set texts to None
    if batch_texts and batch_texts[0] is not None:
        texts = torch.stack(batch_texts, dim=0)
    else:
        texts = None

    coords = torch.stack(padded_coords, dim=0)

    padded_coords_orig = []
    for co in batch_coords_orig:
        co_padded = F.pad(co, (0, 0, 0, max_emb_len - co.shape[0]))
        padded_coords_orig.append(co_padded)

    coords_orig = torch.stack(padded_coords_orig, dim=0)

    pad_mask = embeddings.any(dim=-1)
    pad_lens = (~pad_mask).sum(dim=-1).tolist()
    pad_rate = sum(pad_lens) / len(pad_lens)

    return {
        "labels": batch_labels,
        "embeddings": embeddings,
        "coords": coords,
        "coords_orig": coords_orig,
        "texts": texts,
        "pad_rate": pad_rate,
        "pad_lens": pad_lens,
        "cutoffs": cuttoffs,
        "classes": classes
    }

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def _make_info(dataset, args, *, train=False, sampler=None):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        num_workers=args.workers,
        collate_fn=collate_batch_padded,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        drop_last=train,
    )
    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)
    return DataInfo(loader, sampler)


def create_dataset(args, tokenizer, split_file: str = None):
    def make_dataset(split=None, **kwargs):
        return TextDataset(
            args.embedding_paths,
            getattr(args, "text_path", None),
            tokenizer,
            sample=args.emb_samples,
            split=split,
            split_file=split_file,
            text_id_pattern=getattr(args, "text_id_pattern", None),
            embedding_id_pattern=getattr(args, "embedding_id_pattern", None),
            **kwargs,
        )

    data = {"val": _make_info(make_dataset("test" if split_file else None), args)}

    for name, info in data.items():
        print(f"{name.upper()} DATASET SIZE: {info.dataloader.num_samples}")

    return data