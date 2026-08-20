import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightning
import torch

from classification.model.abmil import ABMIL
from classification.utils.plot import attention_map


def reformat_sid(text: str, pattern: str = None, template: str = "{case}_{specimen}"):
    """
    Reformats an embedding-filename slide id into the report-metadata id format.
    The parsing regex and output template are supplied via configuration
    (`slide_id_pattern` / `slide_id_template`) rather than hardcoded here, since
    the institution-specific slide identifier format is privacy sensitive.
    """
    if not pattern:
        return text

    m = re.search(pattern, text)
    if m is None:
        return text

    return template.format(**m.groupdict())


class BaseMILModel(lightning.LightningModule):
    def __init__(self, dim: int, output_size: int = 1, output: str = None):
        super().__init__()
        self.model = ABMIL(input_size=dim, output_size=output_size, projected_input_size=128, pad_value=None)
        self.output_size = output_size
        self.output = output
        self.create_attention_map = False


class MulticlassMILModel(BaseMILModel):
    def __init__(self, dim: int, output_size: int = 1, output: str = None, slide_id_pattern: str = None,
                 slide_id_template: str = "{case}_{specimen}"):
        super().__init__(dim, output_size, output=output)
        self.slide_path = None
        self.slide_id_pattern = slide_id_pattern
        self.slide_id_template = slide_id_template

    def setup(self, stage: str) -> None:
        self.slide_path = getattr(self.trainer.datamodule.biopsy_dataset, "slide_path", None)

    def predict_step(self, batch, batch_idx):
        slide_ids = batch["slide_id"]
        coords = batch["coords"]
        embedding = batch['embedding']
        logits, attention = self.model(embedding)
        probs = torch.softmax(logits, dim=1)
        probs_list = probs.cpu().numpy().tolist()
        preds = torch.argmax(probs, dim=1)
        preds = [pred.item() for pred in preds]

        if self.create_attention_map:
            with ProcessPoolExecutor(max_workers=32) as executor:
                for s, c, a in zip(slide_ids, coords, attention):
                    executor.submit(attention_map, self.output, self.slide_path + "/" + Path(s).stem + ".tiff",
                                    c.cpu().numpy(),
                                    a.cpu().numpy())

        with open(f"{self.output}/predictions.csv", "a") as f:
            csv_writer = csv.writer(f, delimiter=";")
            for sid, cls, probs_values in zip(slide_ids, preds, probs_list):
                label = self.trainer.datamodule.biopsy_dataset.idx_to_cls[cls]
                sid_formatted = reformat_sid(sid, self.slide_id_pattern, self.slide_id_template)
                csv_writer.writerow([sid_formatted, cls, label] + [f"{val:.5f}" for val in probs_values])

    def on_predict_start(self) -> None:
        self.create_output_dirs()
        with open(f"{self.output}/predictions.csv", "w") as f:
            csv_writer = csv.writer(f, delimiter=";")
            csv_writer.writerow(["Report", "class", "label"] + [f"prob_{i}" for i in range(self.output_size)])

    def create_output_dirs(self):
        output_dirs = ["heatmaps", "histograms", "attention_maps"]
        for od in output_dirs:
            dir = f"{self.output}/{od}"
            if not os.path.exists(dir):
                os.makedirs(dir)
