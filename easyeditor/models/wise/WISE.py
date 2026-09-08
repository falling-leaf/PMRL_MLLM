import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
from .MMD_loss import MMD_loss
import torch.nn as nn
import gc

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if l2_norm.dim() == 1:
        l2_norm = l2_norm.unsqueeze(0)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)

class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.mmd_loss = MMD_loss(kernel_type='cosine')
        self.generate_rephrase_sample = True
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        # if the condition below is True, then it is single-edit
        if not config.sequential_edit:
        # if type(adapter_layer) is not WISEAdapter:
            # 如果 adapter_layer 已经是 WISEAdapter，提取其原始层
            if type(adapter_layer) is WISEAdapter:
                # 使用 original_layer 作为基础层（这是保存的原始层副本）
                base_layer = adapter_layer.original_layer
            else:
                base_layer = adapter_layer
            
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, base_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(base_layer)
            print(f"New weights successfully inserted into {layer}")
        elif type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, *args, **kwargs):
        if not self.config.retrieve:
            adapter = self.get_adapter_layer()
            if hasattr(adapter, 'editing') and not adapter.editing:
                if (not adapter.original_layer.weight.equal(adapter.new_weight)
                        and adapter.editing_total_cnt >= self.config.save_freq):
                    adapter.memory_weight.append(adapter.new_weight)

                if len(adapter.memory_weight) > 0 and adapter.editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(adapter.memory_weight), '!!!!!!')
                    adapter.merge_weight()
        # 1. 如果用户传入 model(batch)
        if len(args) == 1 and isinstance(args[0], dict):
            return self.model(args[0])
        # 2. 如果用户传入 model(batch=batch)
        elif "batch" in kwargs and isinstance(kwargs["batch"], dict):
            batch = kwargs.pop("batch")
            return self.model(**batch, **kwargs)
        # 3. 普通 HuggingFace 风格，如 model(input_ids=..., pixel_values=...)
        else:
            return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(self.get_adapter_layer(), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(self.get_adapter_layer(), "training", True)
        setattr(self.get_adapter_layer(), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(self.get_adapter_layer(), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(tokens, last_prompt_token_loc)

            act_loss = self._cal_activation_loss(self.get_adapter_layer().original_layer_output, self.get_adapter_layer().new_weight_layer_output,
                                                  config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = self._cal_memory_neg_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = self._cal_memory_pos_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            self.get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(self.get_adapter_layer(), "editing", False)
        setattr(self.get_adapter_layer(), "training", False)

        editing_total_cnt = getattr(self.get_adapter_layer(), "editing_total_cnt") + 1
        setattr(self.get_adapter_layer(), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss

    def _cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if config is None:
            config = self.config
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        if hasattr(self.get_adapter_layer(), "wise_edit_activation_count"):
            split = getattr(
                self.get_adapter_layer(),
                "wise_edit_activation_count",
                original_layer_output.shape[0] // 2,
            )
            in_scope_dist = euc(
                original_layer_output[:split], new_weight_layer_output[:split], config
            ).mean()
            out_scope_dist = euc(
                original_layer_output[split:], new_weight_layer_output[split:], config
            ).mean()
            return (
                torch.clamp(out_scope_dist - in_scope_dist + config.gamma, min=0)
                + torch.clamp(out_scope_dist - config.alpha, min=0)
                + torch.clamp(config.beta - in_scope_dist, min=0)
            )
        len_temp = original_layer_output.shape[0] / k - 1
        for i,act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config)
                if (i==k-1):
                    out_scope_dist = euc(original_layer_output[int(i-k):, ...], new_weight_layer_output[int(i-k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i-k):int(i+1-k), ...], new_weight_layer_output[int(i-k):int(i+1-k), ...], config)
                # print("in_scope_dist: ", in_scope_dist)
                # print("out_scope_dist: ", out_scope_dist)
            loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def _cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def _cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)
        
        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")



class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None 

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))], self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6], self.original_layer.weight, self.memory_weight + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], self.original_layer.weight, self.memory_weight[-merge_num:], densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        split = getattr(self, "wise_edit_activation_count", None)
        if split is None:
            in_scope_original = self.original_layer_output[:-1, ...]
            in_scope_new = self.new_weight_layer_output[:-1, ...]
        else:
            in_scope_original = self.original_layer_output[:split, ...]
            in_scope_new = self.new_weight_layer_output[:split, ...]
        in_scope_dist = euc(in_scope_original, in_scope_new, self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias, input.view(-1, input.size(-1)), self.new_weight).view(input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            if not self.config.retrieve:
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                if dist1.item() < threshold and dist2.item() < threshold:
                    layer_out = original_layer_output
                elif dist1.item() > dist2.item():
                    layer_out = layer_output
                else:
                    layer_out = new_weight_layer_output
            else:
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1
                if min_dist.dim() > 0:  
                    min_dist = min_dist.mean()
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                else:
                    layer_out = new_weight_layer_output

                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i]
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                    if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                        layer_out = memory_weight_layer_output
                        min_dist = dist
        return layer_out


class WISEMultimodal(WISE):
    def edit(self, config, multimodal_inputs, text_tokens, ans_token_len, act_mask=None, deact_mask=None):
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in text_tokens.items()}, False])
        last_prompt_token_loc = (text_tokens["labels"] == -100).sum(dim=-1) - 1
        
        setattr(self.get_adapter_layer(), "training", True)
        setattr(self.get_adapter_layer(), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(self.get_adapter_layer(), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)        
        
        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([super().get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len)

            act_loss = super()._cal_activation_loss(super().get_adapter_layer().original_layer_output, super().get_adapter_layer().new_weight_layer_output,
                                                    config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)
            # if self.config.model_name == "blip2":
            #     print(self.model.generate(multimodal_inputs[0]))
            # elif self.config.model_name == "minigpt4":
            #     print(self.model.predict_answers(multimodal_inputs))

            if loss_meter.stop():
                super().get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                super().get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = super()._cal_memory_neg_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = super()._cal_memory_pos_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            super().get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                super()._norm_constraint(self.config.norm_constraint)

            if i == 9 and getattr(self.config, "export_lap_samples", False):
                # 0. 获取生成数量，默认为1
                num_rephrase = 10
                rephrase_samples = []
                # 1. 基础编码（仅需一次）
                # 对embedding层进行扰动的方法
                inputs_embeds, attention_mask, targets = self.model.image_encoding(multimodal_inputs[0])

                # 2. 准备求导环境：Detach -> Clone -> Requires Grad
                embeds_for_grad = inputs_embeds.detach().clone()
                embeds_for_grad.requires_grad_(True)

                # 3. 前向传播计算梯度基础
                outputs = self.model.LLM_forward(embeds_for_grad, attention_mask, targets)

                # =============== 健壮的 Logits 提取 ===============
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0].logits
                else:
                    logits = outputs
                # ==========================================================

                # 4. 计算 Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()

                a = shift_logits.view(-1, shift_logits.size(-1))
                b = shift_labels.view(-1)[-ans_token_len:]
                a = a[-b.size(0):,:]

                loss_fct_inner = torch.nn.CrossEntropyLoss(reduction='sum')
                J_LM = loss_fct_inner(a, b)

                # 5. 反向传播获取基础梯度
                grad_full = torch.autograd.grad(
                    outputs=J_LM,
                    inputs=embeds_for_grad,
                    retain_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )[0]

                if self.config.using_imageembedding:
                    print("using_imageembedding")
                    grad_base = grad_full[:, :32, :]
                else: grad_base = grad_full[:, :-ans_token_len, :]
                base_epsilon = getattr(self.config, 'lap_epsilon', 1e-2)

                epsilon_list = torch.linspace(
                    base_epsilon,
                    base_epsilon * num_rephrase,
                    steps=num_rephrase,
                    device=grad_base.device,
                    dtype=grad_base.dtype
                )

                # 6. 循环生成多个样本
                for i in range(num_rephrase):
                    # 计算扰动 Delta
                    # 注意：如果需要每个样本不同，通常在这里加入随机噪声，例如：
                    # noise = torch.randn_like(grad_base) * some_scale
                    epsilon = epsilon_list[i]
                    grad_norm = torch.norm(grad_base, dim=-1, keepdim=True) + 1e-8
                    delta = (grad_base / grad_norm) * epsilon
                    
                    # 7. 应用扰动并生成样本
                    # 使用 detach() 确保扰动后的输入是从新的计算图开始的
                    if self.config.using_imageembedding:
                        noisy_img_part = inputs_embeds[:, :32, :].detach() + delta.detach()
                        txt_part = inputs_embeds[:, 32:, :].detach()
                    else:
                        noisy_img_part = inputs_embeds[:, :-ans_token_len, :].detach() + delta.detach()
                        txt_part = inputs_embeds[:, -ans_token_len:, :].detach()
                    perturbed_inputs_embeds = torch.cat([noisy_img_part, txt_part], dim=1)
                    
                    # 确保 requires_grad 根据需要开启（如果后续还需要对这个前向过程求导）
                    # perturbed_inputs_embeds.requires_grad_(True) 

                    # 8. 独立前向传播以更新 adapter 状态并获取输出
                    # 每次调用都会进入该次循环所属的独立计算图
                    self.model.LLM_forward(perturbed_inputs_embeds, attention_mask, targets)
                    
                    # 获取当前前向传播捕获的特定层输出
                    perturbed_output = super().get_adapter_layer().new_weight_layer_output
                    rephrase_samples.append(perturbed_output)
                    # 将 rephrase_samples 中的每个 tensor 分别保存到本地文件，文件名按索引递增
                    save_dir = "/root/PMRL_MLLM/saved_rephrase_samples"
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    for idx, tensor in enumerate(rephrase_samples):
                        file_path = os.path.join(save_dir, f"rephrase_sample_{idx:03d}.pt")
                        torch.save(tensor.detach().cpu(), file_path)
                        print(f"Rephrase sample {idx} saved to {file_path}")

        # --- pull out info we want to log from the Wise layer ---
        setattr(self.get_adapter_layer(), "editing", False)
        setattr(self.get_adapter_layer(), "training", False)

        editing_total_cnt = getattr(self.get_adapter_layer(), "editing_total_cnt") + 1
        setattr(self.get_adapter_layer(), "editing_total_cnt", editing_total_cnt)
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            super().get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            super().get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def pmrl_loss(
            self,
            embeddings_list,
            tau_alignment=0.05,
            tau_regularization=0.1,
            alignment_weight=1.0,
            regularization_weight=0.1,
            spectral_alignment=False,
            return_components=False,
        ):
        """View consistency plus weighted anti-collapse regularization.

        The old singular-value classification used class 0 as its target.  The
        first singular value almost always dominated, so cross entropy
        saturated at zero and supplied no useful alignment gradient.  Here the
        alignment term directly measures cosine disagreement between the base
        activation and every LAP view.  The regularizer performs token-level
        instance discrimination on the consensus representation and has its own
        explicit weight so it cannot silently dominate alignment.
        """
        if not isinstance(embeddings_list, list) or len(embeddings_list) < 2:
            raise ValueError("pmrl_loss expects at least two view tensors")
        shapes = [tuple(view.shape) for view in embeddings_list]
        if len(set(shapes)) != 1:
            raise ValueError(f"PMRL view shapes differ: {shapes}")
        views = torch.stack(embeddings_list, dim=1).float()  # [N, V, D]
        if not torch.isfinite(views).all():
            raise FloatingPointError("PMRL input contains NaN/Inf")
        if tau_alignment <= 0 or tau_regularization <= 0:
            raise ValueError("PMRL temperatures must be positive")

        normalized = F.normalize(views, p=2, dim=-1, eps=1e-6)
        if spectral_alignment:
            # Paper-faithful RCSL: variants are rows of H_s [V, D]. Detach the
            # original row as the asymmetric semantic anchor, and minimize the
            # residual spectral energy outside the leading rank-1 direction.
            anchor = normalized[:, :1, :].detach()
            spectral_views = torch.cat([anchor, normalized[:, 1:, :]], dim=1)
            singular_values = torch.linalg.svdvals(spectral_views)
            energy = singular_values.square()
            loss_alignment = (
                energy[:, 1:].sum(dim=1)
                / energy.sum(dim=1).clamp_min(1e-8)
            ).mean() / tau_alignment
        else:
            base = normalized[:, :1, :].detach()
            cosine = (base * normalized[:, 1:, :]).sum(dim=-1)
            loss_alignment = ((1.0 - cosine) / tau_alignment).mean()

        consensus = F.normalize(normalized.mean(dim=1), p=2, dim=-1, eps=1e-6)
        logits_reg = torch.matmul(consensus, consensus.T) / tau_regularization
        reg_targets = torch.arange(consensus.size(0), device=consensus.device)
        loss_regularization = F.cross_entropy(logits_reg, reg_targets)

        # Balance regularization against the live alignment magnitude. This
        # preserves the anti-collapse direction while preventing a raw CE term
        # that is orders of magnitude larger from dominating optimization.
        balanced_regularization = (
            loss_regularization
            / loss_regularization.detach().clamp_min(1e-8)
            * loss_alignment.detach().clamp_min(1e-8)
        )
        total_loss = (
            float(alignment_weight) * loss_alignment
            + float(regularization_weight) * balanced_regularization
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("PMRL loss is non-finite")
        print(
            "loss_alignment: {}, loss_regularization: {}, "
            "alignment_weight: {}, regularization_weight: {}".format(
                loss_alignment, loss_regularization,
                alignment_weight, regularization_weight,
            )
        )
        if return_components:
            return total_loss, {
                "alignment": loss_alignment,
                "regularization": loss_regularization,
            }
        return total_loss

    def mllm_forward(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len, k):
        if self.config.model_name == "blip2" or self.config.model_name == "minigpt4":
            outputs = self.model(multimodal_inputs)
            logits = outputs.logits
            labels = text_tokens["labels"]
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = logits[:-k, :-1, :].contiguous()
            bs = text_tokens["labels"].shape[0]
        else: 
            outputs = self.model(**multimodal_inputs)
            logits = outputs.logits
            labels = []
            # 这里事实上是将acc和loc放在了两个batch上，然后来跑
            shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
            shift_logits = logits[:-k, :-1, :].contiguous()
            bs = text_tokens["input_ids"].shape[0] - k
        return outputs, logits, labels, shift_labels, shift_logits, bs

    def mllm_forward_llava(self, inputs_embeds, labels, ans_token_len):
        """专门为 LLaVA 优化的 forward，直接接收 embeds"""
        outputs = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits
        
        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        bs = labels.shape[0]
        
        return outputs, logits, labels, shift_labels, shift_logits, bs


    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        """Compute baseline WISE loss, optionally augmented by LAP + PMRL.

        Baseline and enhancement are deliberately isolated: baseline always
        performs the canonical edit and locality forwards first.  LAP sampling
        is invoked only when all three enhancement switches are enabled.
        """
        k = self.config.batch_size if hasattr(self.model.config, "batch_size") else 1
        if k != 1:
            raise AssertionError("Not support Batch Edit")

        if self.config.model_name not in ("blip2", "minigpt4"):
            edit_inputs, locality_inputs = multimodal_inputs
            outputs = self.model(**edit_inputs)
            adapter = self.get_adapter_layer()
            edit_original_output = adapter.original_layer_output
            edit_new_output = adapter.new_weight_layer_output
            self.model(**locality_inputs)
            locality_original_output = adapter.original_layer_output
            locality_new_output = adapter.new_weight_layer_output

            shift_labels = edit_inputs["labels"][:, 1:].contiguous()
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            token_loss = CrossEntropyLoss(reduction="none")(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            ).view_as(shift_labels)
            answer_mask = shift_labels.ne(-100)
            if not answer_mask.any():
                raise RuntimeError("HF multimodal edit batch contains no supervised target tokens")
            ft_loss = (token_loss * answer_mask).sum() / answer_mask.sum()
            enhancement_loss = outputs.logits.new_zeros(())
            if (
                getattr(self.config, "using_extra", False)
                and getattr(self.config, "using_lap", False)
                and getattr(self.config, "using_pmrl", False)
            ):
                enhancement_loss = self._compute_hf_lap_pmrl_loss(
                    edit_inputs, edit_new_output
                )

            # LAP forwards overwrite the adapter slots. Install the exact
            # baseline edit/locality pair for the activation-margin loss.
            adapter.original_layer_output = torch.cat(
                [edit_original_output.reshape(-1, edit_original_output.size(-1)),
                 locality_original_output.reshape(-1, locality_original_output.size(-1))],
                dim=0,
            )
            adapter.new_weight_layer_output = torch.cat(
                [edit_new_output.reshape(-1, edit_new_output.size(-1)),
                 locality_new_output.reshape(-1, locality_new_output.size(-1))],
                dim=0,
            )
            adapter.wise_edit_activation_count = (
                edit_original_output.numel() // edit_original_output.size(-1)
            )
            return ft_loss + enhancement_loss

        edit_inputs, locality_inputs = multimodal_inputs
        edit_outputs = self.model(edit_inputs)
        adapter = self.get_adapter_layer()
        edit_original_output = adapter.original_layer_output
        edit_new_output = adapter.new_weight_layer_output

        self.model(locality_inputs)
        locality_original_output = adapter.original_layer_output
        locality_new_output = adapter.new_weight_layer_output

        enhancement_loss = edit_outputs.logits.new_zeros(())
        if (
            getattr(self.config, "using_extra", False)
            and getattr(self.config, "using_lap", False)
            and getattr(self.config, "using_pmrl", False)
        ):
            enhancement_loss = self._compute_lap_pmrl_loss(
                edit_inputs, ans_token_len, edit_new_output
            )

        # Enhancement forwards overwrite adapter captures. Restore the exact
        # edit/locality pair needed by baseline WISE activation separation.
        adapter.original_layer_output = torch.cat(
            [
                edit_original_output.reshape(-1, edit_original_output.size(-1)),
                locality_original_output.reshape(-1, locality_original_output.size(-1)),
            ],
            dim=0,
        )
        adapter.new_weight_layer_output = torch.cat(
            [
                edit_new_output.reshape(-1, edit_new_output.size(-1)),
                locality_new_output.reshape(-1, locality_new_output.size(-1)),
            ],
            dim=0,
        )
        adapter.wise_edit_activation_count = (
            edit_original_output.numel() // edit_original_output.size(-1)
        )

        shift_labels = edit_outputs.labels[:, 1:].contiguous()
        shift_logits = edit_outputs.logits[:, :-1, :].contiguous()
        token_loss = CrossEntropyLoss(reduction="none")(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view(shift_labels.shape[0], -1)
        answer_mask = shift_labels.ne(-100)
        ft_loss = (
            (token_loss * answer_mask).sum(1)
            / answer_mask.sum(1).clamp_min(1)
        ).mean()
        return ft_loss + enhancement_loss

    def _compute_lap_pmrl_loss(self, edit_inputs, ans_token_len, base_output):
        """Generate LAP views and calculate PMRL without filesystem effects."""
        if self.config.model_name not in ("blip2", "minigpt4"):
            raise NotImplementedError("LAP + PMRL is currently validated for BLIP2-style wrappers")

        inputs_embeds, attention_mask, targets = self.model.image_encoding(edit_inputs)
        probe_embeds = inputs_embeds.detach().clone().requires_grad_(True)
        probe_result = self.model.LLM_forward(probe_embeds, attention_mask, targets)
        probe_outputs = probe_result[0] if isinstance(probe_result, tuple) else probe_result
        probe_logits = probe_outputs.logits
        probe_labels = targets[:, 1:].contiguous()
        probe_shift_logits = probe_logits[:, :-1, :].contiguous()
        probe_loss = CrossEntropyLoss(reduction="sum")(
            probe_shift_logits.reshape(-1, probe_shift_logits.size(-1)),
            probe_labels.reshape(-1),
        )
        grad_full = torch.autograd.grad(probe_loss, probe_embeds, retain_graph=False)[0]

        if getattr(self.config, "using_image_embedding", False):
            perturb_end = min(32, inputs_embeds.size(1))
        else:
            perturb_end = max(1, inputs_embeds.size(1) - int(ans_token_len))
        grad_base = grad_full[:, :perturb_end, :]
        grad_direction = grad_base / torch.norm(
            grad_base, dim=-1, keepdim=True
        ).clamp_min(1e-8)

        views = [base_output]
        num_rephrase = int(getattr(self.config, "num_rephrase", 5))
        epsilon = float(getattr(self.config, "lap_epsilon", 1e-3))
        for sample_idx in range(num_rephrase):
            scale = epsilon * float(sample_idx + 1)
            perturbed = inputs_embeds.detach().clone()
            perturbed[:, :perturb_end, :] += grad_direction.detach() * scale
            self.model.LLM_forward(perturbed, attention_mask, targets)
            views.append(self.get_adapter_layer().new_weight_layer_output)

        flattened = [view.reshape(-1, view.size(-1)) for view in views]
        common_length = min(view.size(0) for view in flattened)
        aligned_views = [view[:common_length] for view in flattened]
        return self.pmrl_loss(
            aligned_views,
            tau_alignment=float(self.config.pmrl_tau_alignment),
            tau_regularization=float(self.config.pmrl_tau_regularization),
            alignment_weight=float(getattr(self.config, "pmrl_alignment_weight", 1.0)),
            regularization_weight=float(getattr(self.config, "pmrl_regularization_weight", 0.1)),
            spectral_alignment=bool(getattr(self.config, "pmrl_spectral_alignment", False)),
        ) * float(self.config.pmrl_scale)

    def _build_lar_perturbations(
        self, base_embeds, gradient, mask, num_views, epsilon,
        random_start=False, pgd_steps=1, step_size=None,
    ):
        """Projected adversarial variants with optional random initialization."""
        mask_f = mask.unsqueeze(-1).to(base_embeds.dtype)
        grad = gradient * mask_f
        grad_norm = grad.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        direction = grad / grad_norm.to(grad.dtype).view(-1, 1, 1)
        step_size = float(step_size if step_size is not None else epsilon)
        variants = []
        for view_idx in range(int(num_views)):
            if random_start:
                delta = torch.randn_like(base_embeds) * mask_f
                norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                radius = torch.rand(
                    delta.size(0), device=delta.device, dtype=torch.float32
                ) * float(epsilon)
                delta = delta / norm.to(delta.dtype).view(-1, 1, 1)
                delta = delta * radius.to(delta.dtype).view(-1, 1, 1)
            else:
                # Backward-compatible deterministic sweep across the bounded
                # ray; unlike the old path every view remains inside epsilon.
                delta = direction * (
                    float(epsilon) * float(view_idx + 1) / float(num_views)
                )
            if random_start:
                for _step in range(max(1, int(pgd_steps))):
                    delta = (delta + direction * step_size) * mask_f
                    norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                    factor = (float(epsilon) / norm).clamp(max=1.0)
                    delta = delta * factor.to(delta.dtype).view(-1, 1, 1)
            variants.append(delta.detach())
        return variants

    def _prepare_hf_lap_inputs(self, multimodal_inputs):
        """Build fused embeddings and forward kwargs for supported HF MLLMs."""
        model = self.model
        core = model.model
        input_ids = multimodal_inputs["input_ids"]
        text_embeds = core.get_input_embeddings()(input_ids)
        model_name = self.config.model_name.lower()
        if "qwen2-vl" in model_name:
            image_features = core.get_image_features(
                multimodal_inputs["pixel_values"],
                multimodal_inputs.get("image_grid_thw"),
            )
        elif "llava-onevision" in model_name:
            image_features = core.get_image_features(
                multimodal_inputs["pixel_values"],
                multimodal_inputs["image_sizes"],
                batch_num_images=multimodal_inputs.get("batch_num_images"),
            )
        else:
            raise NotImplementedError(
                f"HF LAP + PMRL does not support model {self.config.model_name}"
            )
        image_features = torch.cat(image_features, dim=0).to(
            text_embeds.device, text_embeds.dtype
        )
        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_features
        )
        fused_embeds = text_embeds.masked_scatter(image_mask, image_features)
        forward_kwargs = {
            "attention_mask": multimodal_inputs["attention_mask"],
            "use_cache": False,
            "return_dict": True,
        }
        if "qwen2-vl" in model_name:
            position_ids, _ = core.get_rope_index(
                input_ids,
                multimodal_inputs.get("image_grid_thw"),
                multimodal_inputs.get("video_grid_thw"),
                multimodal_inputs["attention_mask"],
            )
            forward_kwargs["position_ids"] = position_ids
        return fused_embeds, image_mask.any(dim=-1), forward_kwargs

    def _hf_target_loss(self, logits, labels, reduction):
        """Return target-token CE without promoting the full vocab logits to fp32."""
        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        valid = shift_labels.ne(-100)
        if not valid.any():
            raise RuntimeError("HF LAP probe has no supervised target tokens")
        selected_logits = shift_logits[valid]
        selected_labels = shift_labels[valid]
        if not torch.isfinite(selected_logits).all():
            raise FloatingPointError("HF LAP target logits are non-finite")
        return CrossEntropyLoss(reduction=reduction)(selected_logits, selected_labels)

    def _compute_hf_lap_pmrl_loss(self, multimodal_inputs, base_output):
        """Generate LAP views from fused HF vision/text embeddings."""
        model = self.model
        fused_embeds, visual_token_mask, forward_kwargs = self._prepare_hf_lap_inputs(
            multimodal_inputs
        )
        probe_embeds = fused_embeds.detach().clone().requires_grad_(True)
        probe_outputs = model(inputs_embeds=probe_embeds, **forward_kwargs)
        probe_loss = self._hf_target_loss(
            probe_outputs.logits, multimodal_inputs["labels"], reduction="sum"
        )
        if not torch.isfinite(probe_loss):
            raise FloatingPointError("HF LAP probe loss is non-finite")
        grad_full = torch.autograd.grad(
            probe_loss, probe_embeds, retain_graph=False
        )[0]

        answer_mask = multimodal_inputs["labels"].ne(-100)
        if getattr(self.config, "lar_joint_perturbation", False):
            perturb_mask = (~answer_mask) & multimodal_inputs["attention_mask"].bool()
        elif getattr(self.config, "using_image_embedding", False):
            perturb_mask = visual_token_mask
        else:
            perturb_mask = (~answer_mask) & multimodal_inputs["attention_mask"].bool()
        if not perturb_mask.any():
            raise RuntimeError("HF LAP perturbation mask is empty")
        masked_grad = grad_full * perturb_mask.unsqueeze(-1).to(grad_full.dtype)
        denom = masked_grad.float().flatten(1).norm(dim=1)
        random_start = bool(getattr(self.config, "lar_random_start", False))
        if not torch.isfinite(denom).all():
            raise RuntimeError("LAP gradient is non-finite on selected HF multimodal region")
        if torch.any(denom == 0) and not random_start:
            raise RuntimeError("LAP gradient is zero on selected HF multimodal region")
        direction = masked_grad / denom.clamp_min(1e-8).to(masked_grad.dtype).view(-1, 1, 1)

        views = [base_output.detach()]
        variant_target_losses = []
        epsilon = float(getattr(self.config, "lap_epsilon", 1e-3))
        num_rephrase = int(getattr(self.config, "num_rephrase", 5))
        random_start = bool(getattr(self.config, "lar_random_start", False))
        pgd_steps = int(getattr(self.config, "lar_pgd_steps", 1))
        step_size = float(getattr(self.config, "lar_step_size", epsilon))
        if random_start:
            perturbations = []
            mask_f = perturb_mask.unsqueeze(-1).to(fused_embeds.dtype)
            for _view_idx in range(num_rephrase):
                delta = torch.randn_like(fused_embeds) * mask_f
                delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                radius = torch.rand(
                    delta.size(0), device=delta.device, dtype=torch.float32
                ) * epsilon
                delta = delta / delta_norm.to(delta.dtype).view(-1, 1, 1)
                delta = delta * radius.to(delta.dtype).view(-1, 1, 1)
                # Paper-faithful randomized projected ascent: recompute the
                # adversarial gradient at each independent random start.
                for _step in range(max(1, pgd_steps)):
                    candidate = (
                        fused_embeds.detach() + delta.detach()
                    ).requires_grad_(True)
                    candidate_outputs = model(inputs_embeds=candidate, **forward_kwargs)
                    candidate_loss = self._hf_target_loss(
                        candidate_outputs.logits,
                        multimodal_inputs["labels"],
                        reduction="sum",
                    )
                    candidate_grad = torch.autograd.grad(
                        candidate_loss, candidate, retain_graph=False
                    )[0] * mask_f
                    candidate_norm = candidate_grad.float().flatten(1).norm(
                        dim=1
                    ).clamp_min(1e-8)
                    candidate_direction = candidate_grad / candidate_norm.to(
                        candidate_grad.dtype
                    ).view(-1, 1, 1)
                    delta = (delta + candidate_direction * step_size) * mask_f
                    delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                    factor = (epsilon / delta_norm).clamp(max=1.0)
                    delta = delta * factor.to(delta.dtype).view(-1, 1, 1)
                perturbations.append(delta.detach())
        else:
            perturbations = self._build_lar_perturbations(
                fused_embeds,
                grad_full,
                perturb_mask,
                num_views=num_rephrase,
                epsilon=epsilon,
                random_start=False,
                pgd_steps=1,
                step_size=step_size,
            )
        for delta in perturbations:
            perturbed = fused_embeds.detach() + delta
            variant_outputs = model(inputs_embeds=perturbed, **forward_kwargs)
            views.append(self.get_adapter_layer().new_weight_layer_output)
            variant_target_losses.append(
                self._hf_target_loss(
                    variant_outputs.logits,
                    multimodal_inputs["labels"],
                    reduction="mean",
                )
            )

        shapes = [tuple(view.shape) for view in views]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"HF LAP view shapes differ: {shapes}")
        if not any(
            torch.norm((view - views[0]).float()).item() > 0 for view in views[1:]
        ):
            raise RuntimeError("HF LAP produced identical adapter views")
        flattened = [view.reshape(-1, view.size(-1)) for view in views]
        representation_loss = self.pmrl_loss(
            flattened,
            tau_alignment=float(self.config.pmrl_tau_alignment),
            tau_regularization=float(self.config.pmrl_tau_regularization),
            alignment_weight=float(getattr(self.config, "pmrl_alignment_weight", 1.0)),
            regularization_weight=float(getattr(self.config, "pmrl_regularization_weight", 0.1)),
            spectral_alignment=bool(getattr(self.config, "pmrl_spectral_alignment", False)),
        ) * float(self.config.pmrl_scale)
        target_weight = float(getattr(self.config, "lar_target_loss_weight", 0.0))
        variant_target_loss = torch.stack(variant_target_losses).mean()
        print(
            "lar_variant_target_loss: {}, lar_target_loss_weight: {}".format(
                variant_target_loss, target_weight
            )
        )
        return representation_loss + target_weight * variant_target_loss
