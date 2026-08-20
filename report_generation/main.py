import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from open_clip_train.file_utils import pt_load
from model.logging import setup_logging
from open_clip_train.distributed import broadcast_object, is_master

import logging

from data.dataset import create_dataset
from model.train import evaluate
from model.distributed import init_distributed_device
from utils.config import load_config

from report_generation.model.factory import create_model_and_transforms, get_tokenizer
from report_generation.model.params import parse_args


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    args = parse_args(args)
    config = load_config(args.config)
    for k, v in config.items():
        # Overwrite CLI arguments
        setattr(args, k, v)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([date_str, model_name_safe])

    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup checkpoint / output directories
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.generated_text_path = os.path.join(log_base_path, "texts")
        args.save_embedding_path = os.path.join(log_base_path, "embeddings")
        args.attn_map_path = os.path.join(log_base_path, "attn_maps")
        for dirname in [args.checkpoint_path, args.generated_text_path, args.save_embedding_path, args.attn_map_path]:
            os.makedirs(dirname, exist_ok=True)

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, _, _ = create_model_and_transforms(
        device=device,
        args=args,
        **model_kwargs,
    )

    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(model, fullgraph=True)

    # load a checkpoint's weights for inference
    assert args.resume is not None, "Please provide a checkpoint via --resume / the config's `resume` key."
    checkpoint = pt_load(args.resume, map_location='cpu')
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    logging.info(f"=> loaded checkpoint '{args.resume}'")

    # initialize dataset
    tokenizer = get_tokenizer()
    data = create_dataset(args, tokenizer=tokenizer, split_file=args.split_file)
    assert len(data), 'At least one eval dataset must be specified.'

    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)

    if args.use_bnb_linear is not None:
        from open_clip.utils import convert_int8_model_to_inference_mode
        convert_int8_model_to_inference_mode(model)

    evaluate(model, data, 0, args, tb_writer=None, tokenizer=tokenizer)


if __name__ == "__main__":
    main(sys.argv[1:])
