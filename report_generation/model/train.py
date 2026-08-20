from typing import List
import json
import logging
import os
import random
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import h5py

from report_generation.utils.text_scorer import TextScorer

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.precision import get_autocast


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def unwrap(m):  # handles DDP/compile wrappers
    return getattr(m, "module", m)


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        gen_loss = None
        all_slide_ids = []
        all_image_features, all_text_features = [], []
        ground_truth_texts, generated_texts = [], []
        gt_classes = []
        text_scorer = TextScorer()
        bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, meteor_scores = [], [], [], [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                slide_ids = batch["labels"]
                images = batch["embeddings"].to(device=device, dtype=input_dtype, non_blocking=True)
                texts = batch["texts"]
                coords = batch["coords"].to(device=device, non_blocking=True)

                batch_size = images.shape[0]
                num_samples += batch_size
                log_message = f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"

                image_features = None
                all_slide_ids.extend(slide_ids)
                gt_classes.extend(batch["classes"])
                if texts is not None:
                    texts = texts.to(device=device, non_blocking=True)
                    with autocast():
                        model_out = model(images, texts, coords=coords)
                        image_features = model_out["image_features"]
                        text_features = model_out["text_features"]
                        logit_scale = model_out["logit_scale"]
                        # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                        # however, system RAM is easily exceeded and compute time becomes problematic
                        all_text_features.append(text_features.cpu())
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()


                        labels = torch.arange(batch_size, device=device).long()
                        total_loss = (
                                             F.cross_entropy(logits_per_image, labels) +
                                             F.cross_entropy(logits_per_text, labels)
                                     ) / 2

                        gen_loss = maybe_compute_generative_loss(model_out)

                    cumulative_loss += total_loss * batch_size
                    
                    if is_master(args) and (i % 10) == 0:
                        log_message += f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"

                        if gen_loss is not None:
                            cumulative_gen_loss += gen_loss * batch_size
                            log_message += f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"

                if is_master(args) and epoch % args.eval_frequency == 0:
                    image_latent, text_features, generated_tokens = unwrap_model(model).generate(images,
                                                                                                 coords=coords,
                                                                                                 seq_len=256,
                                                                                                 max_seq_len=256,
                                                                                                 generation_type="top_k",
                                                                                                 sot_token_id=tokenizer.tokenizer.bos_token_id,
                                                                                                 eos_token_id=tokenizer.tokenizer.eos_token_id)

                    if image_features is None:
                        image_features = image_latent

                    generated_text = tokenizer.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    generated_texts.extend(generated_text)

                    if texts is not None:
                        ground_truth_text = tokenizer.tokenizer.batch_decode(texts, skip_special_tokens=False)
                        ground_truth_texts.extend(ground_truth_text)

                        for label, generated in zip(ground_truth_text, generated_text):
                            bleu, rouge1, rouge2, rougel, meteor = text_scorer(label, generated)
                            bleu_scores.append(bleu)
                            rouge1_scores.append(rouge1)
                            rouge2_scores.append(rouge2)
                            rougeL_scores.append(rougel)
                            meteor_scores.append(meteor)

                if image_features is not None:
                    all_image_features.append(image_features.cpu())


                if is_master(args) and (i % 10) == 0:
                    logging.info(log_message)

            # endfor

            if all_image_features and all_text_features:
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                )
                loss = cumulative_loss / num_samples
                metrics.update(
                    {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
                )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

            if bleu_scores:
                metrics.update({"bleu": sum(bleu_scores) / len(bleu_scores)})

            if rouge1_scores and rouge2_scores and rougeL_scores:
                metrics.update({"rouge_1": sum(rouge1_scores) / len(rouge1_scores)})
                metrics.update({"rouge_2": sum(rouge2_scores) / len(rouge2_scores)})
                metrics.update({"rouge_L": sum(rougeL_scores) / len(rougeL_scores)})

            if meteor_scores:
                metrics.update({"meteor": sum(meteor_scores) / len(meteor_scores)})


            if generated_texts:
                assert len(generated_texts) == len(all_slide_ids)
                idx = random.randrange(len(generated_texts))
                print_str = f"ID: {all_slide_ids[idx]}\n\n"

                if ground_truth_texts:
                    assert len(ground_truth_texts) == len(generated_texts)
                    print_str += "Ground truth:\n"
                    print_str += ground_truth_texts[idx] + "\n"

                print_str += "\nGenerated text:\n"
                print_str += generated_texts[idx] + "\n"
                print_str += "-" * 20 + "\n\n"
                print(print_str)

            if args.save_captions:
                save_captions(args.generated_text_path, epoch, all_slide_ids, generated_texts, ground_truth_texts)

            if all_slide_ids and all_image_features:
                save_embeddings(os.path.join(args.save_embedding_path, f"epoch_{epoch}.h5"), all_slide_ids, all_image_features)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def text_metrics(reference: str, hypothesis: str, show=False) -> tuple[float, float, float, float, float]:
    """
    Computes BLEU, ROUGE, and METEOR scores for a given reference and hypothesis text.
    """

    # Tokenize reference and hypothesis for BLEU & METEOR
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    # 1. Compute BLEU Score
    smoothing = SmoothingFunction().method1  # Avoids zero scores for short texts
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

    # 2. Compute ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, hypothesis)

    # 3. Compute METEOR Score
    meteor = meteor_score([ref_tokens], hyp_tokens)

    return bleu, rouge['rouge1'].fmeasure, rouge['rouge2'].fmeasure, rouge['rougeL'].fmeasure, meteor


def save_captions(path, epoch, slide_ids, generated, ground_truth = ()):
    epoch_path = os.path.join(path, f"epoch_{epoch}")
    if not os.path.exists(epoch_path):
        os.mkdir(epoch_path)

        generated_path = os.path.join(epoch_path, "generated")
        if not os.path.exists(generated_path):
            os.mkdir(generated_path)

        for label, generated in zip(slide_ids, generated):
            with open(os.path.join(generated_path, f"{label}.txt"), "w") as f:
                f.write(generated)

    with open(os.path.join(path, f"epoch_{epoch}.txt"), "w") as f:
        if ground_truth:
            for label, gt, generated in zip(slide_ids, ground_truth, generated):
                f.write(f"ID: {label}\n\n")
                f.write("Ground truth:\n")
                f.write(f"{gt}\n\n")
                f.write("Generated:\n")
                f.write(f"{generated}\n\n\n")
                f.write("-" * 20 + "\n\n")
        else:
            for label, generated in zip(slide_ids, generated):
                f.write(f"ID: {label}\n\n")
                f.write("Generated:\n")
                f.write(f"{generated}\n")
                f.write("-" * 20 + "\n\n")


def save_embeddings(file: str, labels: List[str], embeddings: List[torch.Tensor]):
    embeddings: npt.NDArray = torch.cat(embeddings, dim=0).numpy()

    with h5py.File(file, "w") as f:
        f["labels"] = labels
        f["embeddings"] = embeddings

