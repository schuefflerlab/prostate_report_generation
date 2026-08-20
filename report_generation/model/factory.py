import json
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path

import torch
from open_clip.transform import AugmentationCfg
from transformers import AutoTokenizer

from report_generation.model.slide_model import SlideModel
from report_generation.data.tokenizer import Tokenizer
from report_generation.model.transformer import CLIPTextCfg, MultimodalCfg, CLIPEmbeddingCfg


MODEL_CONFIG_PATH = Path(__file__).parents[1] / "config" / "model"


def create_model(device, args):
    """
    :param device: Device ('cpu', 'cuda', ...).
    :return: The created model instance.
    """

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = load_model_config(args.model_name)
    text_cfg = CLIPTextCfg(**model_cfg["text_cfg"])
    case_cfg = CLIPEmbeddingCfg(**model_cfg["case_cfg"])
    multimodal_cfg = MultimodalCfg(**model_cfg["multimodal_cfg"])

    model = SlideModel(
        multimodal_cfg=multimodal_cfg,
        text_cfg=text_cfg,
        embedding_cfg=case_cfg
    )

    if isinstance(device, str):
        device = torch.device(device)

    return model.to(device)


def create_model_and_transforms(
        device: Union[str, torch.device] = 'cpu',
        args = None,
        **model_kwargs,
):
    model = create_model(device=device, args=args)

    return model, None, None


def get_tokenizer(tokenizer: str = "trained"):
    match tokenizer:
        case "trained":
            tokenizer = Tokenizer("./german-tokenizer-pre")
        case _:
            raise NotImplementedError

    return tokenizer

def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def load_model_config(name):
    fn = MODEL_CONFIG_PATH / f"{name}.json"
    with open(fn) as model_file:
        config = json.load(model_file)

    return config




