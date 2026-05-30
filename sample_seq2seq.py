"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(
        split='valid',
        clamp_step=0,
        seed2=105,
        clip_denoised=False,
        save_intermediate=True,
    )
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def _sampling_timesteps(diffusion, use_ddim, gap):
    indices = list(range(diffusion.num_timesteps))[::-1]
    if use_ddim:
        return indices[::gap]
    return indices


def _decode_target_texts(model, tokenizer, latent, input_ids_mask_ori, seq_len):
    logits = model.get_logits(latent)
    cands = th.topk(logits, k=1, dim=-1)

    decoded = []
    for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
        len_x = seq_len - sum(input_mask).tolist()
        decoded.append(tokenizer.decode_token(seq[len_x:]))
    return decoded


def _decode_source_and_reference(tokenizer, input_ids_x, input_ids_mask_ori, seq_len):
    sources = []
    references = []

    for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
        len_x = seq_len - sum(input_mask).tolist()
        sources.append(tokenizer.decode_token(seq[:len_x]))
        references.append(tokenizer.decode_token(seq[len_x:]))

    return sources, references


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    intermediate_out_path = out_path.replace('.json', '.intermediate.json')
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:

        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask_ori = cond.pop('input_mask')
        input_ids_mask_dev = input_ids_mask_ori.to(dist_util.dev())

        noise_main = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask_dev.unsqueeze(dim=-1), x_start.shape)
        x_noised = th.where(input_ids_mask == 0, x_start, noise_main)
        #temp change to test the ability of model to paraphrase regardless of target length 
        """input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        input_ids_mask_ori = cond.pop('input_mask')             

        src_lens = (input_ids_mask_ori == 0).sum(dim=1)          

        new_mask = th.ones_like(input_ids_mask_ori)             
        for b in range(input_ids_mask_ori.shape[0]):
            new_mask[b, :src_lens[b]] = 0                       

        for b in range(input_ids_mask_ori.shape[0]):
            input_ids_x[b, src_lens[b]:] = tokenizer.pad_token_id

        x_start = model.get_embeds(input_ids_x)
        input_ids_mask_ori = new_mask
        input_ids_mask_dev = new_mask.to(dist_util.dev())

        noise_main = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask_dev.unsqueeze(dim=-1), x_start.shape)
        x_noised = th.where(input_ids_mask == 0, x_start, noise_main)"""
        #temp change to test the ability of model to paraphrase regardless of target length 

        model_kwargs = {}
        if getattr(model, "ecc_mode", False):
            aux_states = []
            for _ in range(model.ecc_num_aux_copies):
                aux_noise = th.randn_like(x_start)
                aux_states.append(th.where(input_ids_mask == 0, x_start, aux_noise))
            model_kwargs = {
                "input_mask": input_ids_mask_dev,
                "aux_states": aux_states,
            }

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        progressive_sample_fn = (
            diffusion.p_sample_loop_progressive
            if not args.use_ddim
            else diffusion.ddim_sample_loop_progressive
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        sampling_timesteps = _sampling_timesteps(diffusion, args.use_ddim, step_gap)

        progressive_kwargs = dict(
            model=model,
            shape=sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            mask=input_ids_mask,
            x_start=x_start,
        )
        if args.use_ddim:
            progressive_kwargs["gap"] = step_gap
        else:
            progressive_kwargs["top_p"] = args.top_p
            progressive_kwargs["clamp_step"] = args.clamp_step
            progressive_kwargs["clamp_first"] = True

        stepwise_recoveries = [[] for _ in range(x_start.shape[0])]
        final_sample = None

        for step_index, (timestep, step_out) in enumerate(
            zip(sampling_timesteps, progressive_sample_fn(**progressive_kwargs))
        ):
            final_sample = step_out["sample"]
            if args.save_intermediate:
                decoded_step = _decode_target_texts(
                    model,
                    tokenizer,
                    step_out["pred_xstart"],
                    input_ids_mask_ori,
                    args.seq_len,
                )
                for example_idx, recover_text in enumerate(decoded_step):
                    stepwise_recoveries[example_idx].append(
                        {
                            "step_index": step_index,
                            "timestep": int(timestep),
                            "recover": recover_text,
                        }
                    )

        if final_sample is None:
            raise RuntimeError("Sampling produced no outputs.")

        word_lst_recover = _decode_target_texts(
            model,
            tokenizer,
            final_sample,
            input_ids_mask_ori,
            args.seq_len,
        )
        word_lst_source, word_lst_ref = _decode_source_and_reference(
            tokenizer,
            input_ids_x,
            input_ids_mask_ori,
            args.seq_len,
        )

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
                
                #version for repeated target for baseline 
                """chunk_out_path = out_path.replace('.json', '.targets.json')
                with open(chunk_out_path, 'a') as fchunk:
                    for i in range(len(word_lst_source)):
                        src_len_i = (input_ids_mask_ori[i] == 0).sum().item()
                        target_latent_i = final_sample[i, src_len_i:, :]

                        chunk_len = target_latent_i.shape[0] // 3
                        chunks_i = [
                            target_latent_i[:chunk_len, :],
                            target_latent_i[chunk_len:2*chunk_len, :],
                            target_latent_i[2*chunk_len:3*chunk_len, :],
                        ]
                        avg_chunk_i = th.stack(chunks_i, dim=0).mean(dim=0)

                        def decode_latent_single(chunk):
                            logits = model.get_logits(chunk.unsqueeze(0))
                            indices = th.topk(logits, k=1, dim=-1).indices.squeeze()
                            return tokenizer.decode_token(indices)

                        print(json.dumps({
                            "source":  word_lst_source[i],
                            "target1": decode_latent_single(chunks_i[0]),
                            "target2": decode_latent_single(chunks_i[1]),
                            "target3": decode_latent_single(chunks_i[2]),
                            "avg":     decode_latent_single(avg_chunk_i),
                        }), file=fchunk)"""
                #version for repeated target for baseline 


                if args.save_intermediate:
                    fout_intermediate = open(intermediate_out_path, 'a')
                    for (recoveries, ref, src) in zip(stepwise_recoveries, word_lst_ref, word_lst_source):
                        print(
                            json.dumps(
                                {
                                    "source": src,
                                    "reference": ref,
                                    "recover_steps": recoveries,
                                }
                            ),
                            file=fout_intermediate,
                        )
                    fout_intermediate.close()
            dist.barrier()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')
    if args.save_intermediate:
        print(f'### Written intermediate reconstructed texts to {intermediate_out_path}')


if __name__ == "__main__":
    main()
