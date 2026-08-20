import argparse
from pathlib import Path

from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers, AddedToken, processors, decoders
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast


def train_from_scratch(dataset_path: str):
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC()
    ])

    tokenizer.pre_tokenizer = tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True)
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<start_of_text> $A <end_of_text>",
        pair="<start_of_text>:0 $A:0 <end_of_text>:0 <start_of_text>:1 $B:1 <end_of_text>:1",  # Common for pairs
        special_tokens=[
            ("<start_of_text>", 1),
            ("<end_of_text>", 2),
        ],
    )

    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=False,  # Match your pre_tokenizer setting
        trim_offsets=True,
        use_regex=True
    )

    special_tokens_list = [
        AddedToken("<pad>", normalized=False, special=True),  # ID 0
        AddedToken("<start_of_text>", normalized=False, special=True),  # ID 1
        AddedToken("<end_of_text>", normalized=False, special=True),  # ID 2
        AddedToken("<SECTION:MICROSCOPY>", normalized=False, special=True),
        AddedToken("</SECTION:MICROSCOPY>", normalized=False, special=True),
        AddedToken("<NO_MICROSCOPY>", normalized=False, special=True),
        AddedToken("<SECTION:FINDINGS>", normalized=False, special=True),
        AddedToken("</SECTION:FINDINGS>", normalized=False, special=True),
        AddedToken("<NO_FINDINGS>", normalized=False, special=True),
        AddedToken("<GRADE>", normalized=False, special=True, lstrip=False, rstrip=False),
        AddedToken("</GRADE>", normalized=False, special=True, lstrip=False, rstrip=False),
    ]

    additional_clinical_tokens = ["7a", "7b"]
    for tok in additional_clinical_tokens:
        special_tokens_list.append(AddedToken(tok, normalized=False))

    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        special_tokens=special_tokens_list
    )

    files = [str(file) for file in Path(dataset_path).glob("**/*.txt")]
    tokenizer.train(files, trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<start_of_text>",
        eos_token="<end_of_text>",
    )

    fast_tokenizer.save_pretrained("report-tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    train_from_scratch(args.dataset)
