import time

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 512
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev, base_model = None, reference_matrix = None, safety_vector = None,):
    base_weight_list = None
    if base_model != None:
        base_weight_list = []
        base_model.model.layers

        if args.true_sequential:
            finetune_module = ["self_attn.v_proj", "self_attn.q_proj"]
        else:
            finetune_module = ["self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]
        
        
        for layer in base_model.model.layers:
            full = find_layers(layer)    
            base_weight = {}
            for module in finetune_module:
                # import pdb; pdb.set_trace()
                base_weight[module] = full[module].weight.data
            base_weight_list.append(base_weight)

        pass
        del base_model

    ## compute mask
    if reference_matrix != None:
        # import pdb; pdb.set_trace()
        mask_list = [{} for _ in range(len(model.model.layers))]
        if type(reference_matrix) == list:
            mask_list = None
            fim_score_list = reference_matrix
            pass
        else:
            mask_list = None
            fim_score_list = [{} for _ in range(len(model.model.layers))]
            for name, fim_score in reference_matrix.items():
                for module in finetune_module:
                    if module in name:
                        if "self_attn" in name:
                            idx = int(name.split("model.layers.")[-1].split(".self_attn.")[0])
                        else:
                            idx = int(name.split("model.layers.")[-1].split(".mlp.")[0])

                        fim_score_list[idx][module] = fim_score

        del reference_matrix
    if safety_vector != None:
        # import pdb; pdb.set_trace()
        mask_list = [{} for _ in range(len(model.model.layers))]
        if type(safety_vector) == list:
            mask_list = None
            safety_vector_list = safety_vector
            pass
        else:
            mask_list = None
            safety_vector_list = [{} for _ in range(len(model.model.layers))]
            for name, fim_score in safety_vector.items():
                for module in finetune_module:
                    if module in name:
                        if "self_attn" in name:
                            idx = int(name.split("model.layers.")[-1].split(".self_attn.")[0])
                        else:
                            idx = int(name.split("model.layers.")[-1].split(".mlp.")[0])
                        safety_vector_list[idx][module] = fim_score

        del safety_vector
    else:
        mask_list = None
        safety_vector_list = None
        # import pdb; pdb.set_trace()
    ##
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    attention_masks = torch.zeros(
        (args.nsamples, model.seqlen, model.seqlen), dtype=dtype, device=dev
    )
    attention_masks = torch.zeros(
        (args.nsamples, model.seqlen, model.seqlen), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": torch.tensor(range(0, model.seqlen), dtype=torch.int64), "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # import pdb; pdb.set_trace()
            inps[cache["i"]] = inp
            if kwargs.get("attention_mask") != None:
                attention_masks[cache["i"]] = kwargs["attention_mask"]
            else:
                attention_masks[cache["i"]] = torch.ones(
                    (model.seqlen, model.seqlen), dtype=dtype, device=dev
                )
            cache["i"] += 1
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    from tqdm import tqdm
    for batch in tqdm(dataloader, total=len(dataloader)):
        # print("batch[0]:{}".format(batch[0].unsqueeze(0)))
        # print("mask:{}".format(batch[2].unsqueeze(0)))
        # import pdb; pdb.set_trace()
        try:
            # model(batch[0].to(dev))
            model(batch[0].unsqueeze(0).to(dev), attention_mask=batch[2].unsqueeze(0).to(dev))

        except ValueError:
            pass
    # import pdb; pdb.set_trace()
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache["attention_mask"]
    # import pdb; pdb.set_trace()
    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        # layer_hessian = {}
        if args.true_sequential:
            sequential = [
                # ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                # ["self_attn.o_proj"],
                # ["mlp.up_proj", "mlp.gate_proj"],
                # ["mlp.down_proj"],
                ["self_attn.v_proj", "self_attn.q_proj"],
                # ["self_attn.o_proj"],
                # ["mlp.up_proj", "mlp.gate_proj"],
                # ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        # import pdb; pdb.set_trace()
        for names in sequential:
            subset = {n: full[n] for n in names}
            # import pdb; pdb.set_trace()
            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # import pdb; pdb.set_trace()
                outs[j] = layer(inps[j].unsqueeze(0), 
                                attention_mask=attention_masks[j].unsqueeze(0).unsqueeze(0), 
                                position_ids=cache['position_ids'], 
                                position_embeddings=cache['position_embeddings'], 
                                )[0]
            for h in handles:
                h.remove()

            # import pdb; pdb.set_trace()
            
            for name in subset:
                print(i, name)
                # if i == 31 and name == "self_attn.q_proj":
                #     import pdb; pdb.set_trace()
                print("Pruning ...")
                # layer_hessian[name] = gpts[name].H
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    base_W = base_weight_list[i][name].to(dev),
                    mask = mask_list[i][name] if mask_list != None else None,
                    FIM_score = fim_score_list[i][name].to(dev) if fim_score_list != None else None,
                    safety_vector = safety_vector_list[i][name].to(dev) if safety_vector_list != None else None,
                    decorate = args.recalibrate,
                    method = args.method,
                    # alpha = args.alpha,
                    remove_more = args.remove_more,
                )
                gpts[name].free()

        for j in range(args.nsamples):
            try:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).unsqueeze(0), 
                                position_ids=cache['position_ids'], 
                                position_embeddings=cache['position_embeddings'], 
                                )[0]
            except Exception as e:
                print(e)
                # import pdb; pdb.set_trace()
            # if torch.any(outs[j].isnan()) == True:
            #     import pdb; pdb.set_trace()
        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # model_hessian.append(layer_hessian)
        # del layer_hessian

    model.config.use_cache = use_cache

    # print("hessian information matrix will be saved at: {}".format(save_name))
    # torch.save(model_hessian, save_name)
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="LlaMA model to load")
    parser.add_argument("--base_model", type=str, help="base LlaMA model to load", default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        # choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=4096,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default=None, help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--safe_FIM_path", type=str, required=False, default=None,
    )
    parser.add_argument(
        "--safety_vector", type=str, required=False, default=None,
    )

    parser.add_argument(
        "--recalibrate", type=int, required=False, default=1,
    )
    parser.add_argument(
        "--remove_more", type=int, required=False, default=0,
    )
    parser.add_argument(
        "--need_system_prompt", type=int, required=False, default=1,
    )
    parser.add_argument(
        "--method", type=str, required=False, default="IRR",
    )

    # parser.add_argument("--alpha", type=float, default=0.5, help="alpha")

    args = parser.parse_args()
    # print(args)
    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")
    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()
    model.to(DEV)
    if args.safe_FIM_path != None:
        reference_matrix = torch.load(args.safe_FIM_path)
    else:
        reference_matrix = None

    if args.safety_vector != None:
        safety_vector = torch.load(args.safety_vector)
    else:
        safety_vector = None
        # mask
    if args.base_model !=None:
        base_model = get_llama(args.base_model)
    else:
        base_model = None

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, need_system_prompt = args.need_system_prompt
    )
    
    if args.sparsity == 0.0:
        args.sparsity = "Remove_All"
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        llama_sequential(model, dataloader, DEV, base_model=base_model, reference_matrix = reference_matrix, safety_vector = safety_vector)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)

    if args.save:
        save_name = args.save
    else:
        model_name = args.model.split("/")[-1]
        if args.true_sequential == True:
            targets = "q_proj_v_proj"
            pass
        else:
            targets = "all"
            pass
        method_name = args.method
        if args.remove_more == True:
            method_name = method_name + "_more"
        save_name = f'{model_name}_{method_name}_sparsity_{args.sparsity}_{targets}_blocksize_{args.blocksize}'

        save_name = "saved_models/" + save_name
    print("model will be saved at: {}".format(save_name))
    model.save_pretrained(save_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(save_name)
