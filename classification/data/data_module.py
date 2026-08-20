import lightning as L
from torch.utils.data import DataLoader

from classification.data.biopsy_dataset import BiopsyGleasonDataset, pad_collate_bags


class BiopsyDataModule(L.LightningDataModule):
    def __init__(self, feature_path: str, output: str, slide_path: str = None, filename_template: str = None):
        super().__init__()
        self.feature_path = feature_path
        self.output = output
        self.class_names = None
        self.slide_path = slide_path
        self.filename_template = filename_template

        print(self.slide_path)

    def setup(self, stage: str):
        self.biopsy_dataset = BiopsyGleasonDataset(metadata=None, feature_path=self.feature_path,
                                                    slide_path=self.slide_path,
                                                    filename_template=self.filename_template)
        self.class_names = list(self.biopsy_dataset.cls_to_idx.keys())

    def predict_dataloader(self):
        return DataLoader(self.biopsy_dataset, batch_size=32, num_workers=8, collate_fn=pad_collate_bags)
