"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )
    
    # ===== DIAGNOSTIC STEP 1: Check DataLoader output =====
    logger.log("### DIAGNOSTIC: Inspecting DataLoader batch...")
    batch = next(data)
    arr, out_kwargs = batch
    
    logger.log(f"arr shape: {arr.shape}")
    logger.log(f"out_kwargs keys: {out_kwargs.keys()}")
    logger.log(f"input_ids shape: {out_kwargs['input_ids'].shape}")
    logger.log(f"input_mask shape: {out_kwargs['input_mask'].shape}")
    logger.log(f"target_len type: {type(out_kwargs['target_len'])}")
    logger.log(f"target_len shape: {out_kwargs['target_len'].shape if hasattr(out_kwargs['target_len'], 'shape') else 'N/A'}")
    logger.log(f"target_len values: {out_kwargs['target_len']}")
    logger.log(f"target_len dtype: {out_kwargs['target_len'].dtype if hasattr(out_kwargs['target_len'], 'dtype') else type(out_kwargs['target_len'][0])}")
    
    # ===== DIAGNOSTIC STEP 3: Check mask calculation =====
    import torch as th
    out_kwargs_copy = {k: v for k, v in out_kwargs.items()}
    input_ids_mask = th.tensor(out_kwargs_copy['input_mask']).float()
    target_len = th.tensor(out_kwargs_copy['target_len'])
    
    logger.log("\n### DIAGNOSTIC: Checking mask calculation...")
    logger.log(f"input_ids_mask shape: {input_ids_mask.shape}")
    logger.log(f"target_len shape: {target_len.shape}")
    
    context_len = (input_ids_mask == 0).sum(dim=-1)
    logger.log(f"context_len: {context_len}")
    logger.log(f"target_len: {target_len}")
    
    B, S = input_ids_mask.shape
    positions = th.arange(S).unsqueeze(0).expand(B, -1)
    cl = context_len.unsqueeze(1)
    tl = target_len.unsqueeze(1)
    
    t1_mask = (positions >= cl) & (positions < cl + tl)
    t2_mask = (positions >= cl + tl) & (positions < cl + 2 * tl)
    t3_mask = (positions >= cl + 2 * tl) & (positions < cl + 3 * tl)
    
    logger.log(f"t1_mask sum per sample: {t1_mask.sum(dim=-1)}")
    logger.log(f"t2_mask sum per sample: {t2_mask.sum(dim=-1)}")
    logger.log(f"t3_mask sum per sample: {t3_mask.sum(dim=-1)}")
    total_target = t1_mask.sum(dim=-1) + t2_mask.sum(dim=-1) + t3_mask.sum(dim=-1)
    logger.log(f"Total target positions: {total_target}")
    
    # ===== DIAGNOSTIC STEP 4: Check sequence structure =====
    logger.log("\n### DIAGNOSTIC: Checking sequence structure...")
    input_ids_sample = out_kwargs['input_ids'][0]
    input_mask_sample = out_kwargs['input_mask'][0]
    target_len_sample = out_kwargs['target_len'][0]
    
    logger.log(f"First 30 input_ids: {input_ids_sample[:30]}")
    logger.log(f"First 30 input_mask: {input_mask_sample[:30]}")
    logger.log(f"target_len for sample 0: {target_len_sample}")
    
    src_len = (th.tensor(input_mask_sample) == 0).sum().item()
    logger.log(f"Source length (mask=0): {src_len}")
    sep_id = 102  # [SEP] for BERT
    if src_len < len(input_ids_sample):
        logger.log(f"Token at position {src_len}: {input_ids_sample[src_len]} (expecting {sep_id} for [SEP])")
    
    logger.log("### Diagnostic complete. Check output above.\n")
    # ===== END DIAGNOSTICS =====

    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.to(dist_util.dev())
    
    # ... rest of training code ...
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
