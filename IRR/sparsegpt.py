import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # if torch.any(inp.isnan()) == True:
        #     import pdb; pdb.set_trace()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        pre_inp = inp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # if torch.any(inp.isnan()) == True:
        #     import pdb; pdb.set_trace()
        self.H += inp.matmul(inp.t())



    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, base_W = None, mask = None, FIM_score= None, safety_vector = None, 
        decorate = True, method = "IRR", remove_more = False,
    ):
        if base_W == None:
            W = self.layer.weight.data.clone()
        else:
            # pass
            W = self.layer.weight.data.clone() - base_W

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        # H = FIM_score
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        # import pdb; pdb.set_trace()
        try:
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except:
            pass
            import pdb; pdb.set_trace()
        # mask = None
        compute_method = {
            method: True,
        }
        if sparsity == "Remove_All":
            sparsity = 0.0
        unmask_p = [0.00, sparsity]
        
        total_mask_count = 0
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = None
            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    pass
                    if compute_method.get("magnitude", None) != None:
                        compute_method['magnitude'] = torch.abs(W1)
                    if compute_method.get("sensitivity", None) != None:
                        # compute_score['sensitivity'] = W1 ** 2 / (2.0 * torch.diag(Hinv1).reshape((1, -1)))
                        task_fim_score1 = safety_vector[: ,i1:i2]
                        # import pdb; pdb.set_trace()
                        compute_method['sensitivity'] = (W1 ** 2) * task_fim_score1 / 2.0 
                    if compute_method.get("sparsegpt", None) != None:
                        compute_method['sparsegpt'] = W1 ** 2 / (2.0 * torch.diag(Hinv1).reshape((1, -1)))
                    if compute_method.get("IRR", None) != None:
                        # print("compute safe_fisher")
                        fim_score1 = FIM_score[: ,i1:i2]
                        safety_vector_score1 = safety_vector[: ,i1:i2]

                        # import pdb; pdb.set_trace()
                        task_sign = torch.sign(W1)
                        safe_sign = torch.sign(safety_vector_score1)
                        not_matching_positions = task_sign != safe_sign

                        if remove_more == True:
                            fim_score1 = torch.where(not_matching_positions, fim_score1, fim_score1 * -1.0)
                            not_matching_positions = torch.ones_like(W1, dtype=torch.bool)
                            
                        full_condition_mask = torch.zeros_like(W1, dtype=torch.bool)
                        safety_score = torch.zeros_like(W1)

                        safety_score[not_matching_positions] = fim_score1[not_matching_positions]

                        # 升序排列
                        # tick = time.time()
                        sorted_values, sorted_indices = torch.sort(safety_score[not_matching_positions], stable=True)
                        # total_sort_time += time.time() - tick
                        if type(unmask_p) == list:
                            # mask1 = torch.zeros_like(safety_score[not_matching_positions], dtype=torch.bool)
                            mask2 = torch.zeros_like(safety_score[not_matching_positions], dtype=torch.bool)
                            # mask1[sorted_indices[:int(sorted_values.numel() * unmask_p[0])]] = 1
                            mask2[sorted_indices[int(sorted_values.numel() * unmask_p[1]):]] = 1
                            # 条件掩码计算位置 condition_mask 
                            # condition_mask = mask1 | mask2
                            # condition_mask = mask2
                            full_condition_mask[not_matching_positions] = mask2
                            # full_condition_mask[not_matching_positions] = condition_mask
                            mask1 = full_condition_mask
                            total_mask_count += mask1.sum()
                            # print("mask1 p = {:.2f}%".format(mask1.sum() * 100.00 / mask1.numel()))

                    if compute_method.get("IRR_wo_sign", None) != None:
                        fim_score1 = FIM_score[: ,i1:i2]
                        safety_vector_score1 = safety_vector[: ,i1:i2]

                        # task_sign = torch.sign(W1)
                        # safe_sign = torch.sign(safety_vector_score1)
                        not_matching_positions = torch.ones_like(fim_score1, dtype=torch.bool)

                        full_condition_mask = torch.zeros_like(W1, dtype=torch.bool)
                        safety_score = torch.zeros_like(W1)

                        safety_score[not_matching_positions] = fim_score1[not_matching_positions]

                        # 升序排列
                        sorted_values, sorted_indices = torch.sort(safety_score[not_matching_positions], stable=True)

                        if type(unmask_p) == list:
                            mask1 = torch.zeros_like(safety_score[not_matching_positions], dtype=torch.bool)
                            mask2 = torch.zeros_like(safety_score[not_matching_positions], dtype=torch.bool)
                            mask1[sorted_indices[:int(sorted_values.numel() * unmask_p[0])]] = 1
                            mask2[sorted_indices[int(sorted_values.numel() * unmask_p[1]):]] = 1
                            # 条件掩码计算位置 condition_mask 
                            condition_mask = mask1 | mask2
                            full_condition_mask[not_matching_positions] = condition_mask
                            mask1 = full_condition_mask
                            total_mask_count += mask1.sum()
                            
                    if compute_method.get("random_mask", None) != None:
                        p = 1.0 - unmask_p[1]
                        mask1 = torch.rand(W1.shape) < p
                        total_mask_count += mask1.sum()
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                # import pdb; pdb.set_trace()
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                # import pdb; pdb.set_trace()
                if decorate == True:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            
            if decorate == True:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            
            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))
            
            

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        count_mask_p = total_mask_count / W.numel()
        print("mask p={:.2f}%".format(count_mask_p * 100.0))
        if base_W == None:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = (base_W + W.reshape(self.layer.weight.shape)).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        # self.layer.weight.data = self.layer.weight.data.to(torch.float16)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
