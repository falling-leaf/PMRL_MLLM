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
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
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
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
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
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        # print("Input dim:", self.get_adapter_layer().weight.shape[1])
        # print("Output dim:", self.get_adapter_layer().weight.shape[0])
        # print("original_layer_output_shape: ", original_layer_output.shape)
        # print("new_weight_layer_output: ", new_weight_layer_output.shape)
        if self.config.model_name == "blip2":
            original_layer_output = original_layer_output.reshape(2, -1, original_layer_output.size(-1))
            new_weight_layer_output = new_weight_layer_output.reshape(2, -1, new_weight_layer_output.size(-1))
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
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
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
        
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
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

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
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
            lambda_reg=1.0
        ):
        if not isinstance(embeddings_list, list):
            raise ValueError("pmrl_loss expects a list of tensors.")
            
        # 1. 堆叠
        Z = torch.stack(embeddings_list, dim=2) 
        
        # =======================================================
        # 【关键修复】: 强制转换为 float32 以避免 SVD 崩溃
        # =======================================================
        original_dtype = Z.dtype
        Z = Z.to(torch.float32)

        # 2. 模态内标准化 (Normalization)
        # 增加 eps 防止除以零 (虽然 F.normalize 默认有 eps，但手动指定更稳)
        Z = F.normalize(Z, p=2, dim=1, eps=1e-6)
        
        # 3. SVD
        # 此时 Z 是 float32，SVD 会非常稳定
        try:
            U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        except RuntimeError as e:
            # 即使在 float32 下，如果输入全是 NaN 也会挂，这里做个兜底
            print(f"SVD failed with error: {e}")
            # 打印一下 Z 的统计信息帮助 debug
            print(f"Z stats - Max: {Z.max()}, Min: {Z.min()}, IsNaN: {torch.isnan(Z).any()}")
            return torch.tensor(0.0, device=Z.device, requires_grad=True)

        batch_size = Z.shape[0] 
        device = Z.device

        # --- Alignment Loss ---
        alignment_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss_alignment = F.cross_entropy(S / tau_alignment, alignment_targets)

        # --- Regularization Loss ---
        u1 = U[:, :, 0] # (N, D)
        
        # 矩阵乘法也在 float32 下进行，精度更高
        logits_reg = torch.matmul(u1, u1.T) / tau_regularization
        
        reg_targets = torch.arange(batch_size, dtype=torch.long, device=device)
        loss_regularization = F.cross_entropy(logits_reg, reg_targets)

        # 4. 总损失
        print("loss_alignment: {}, loss_regularization: {}".format(loss_alignment, loss_regularization))
        total_loss = loss_alignment + lambda_reg * loss_regularization
        
        # 如果外部需要 float16 的 loss (通常不需要，因为 scalar loss 会自动适配)，可以 cast 回去
        # 但通常建议保持 float32 直到 backward
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
            # 这里事实上是将acc和loc放在了两个batch上，然后来跑
            shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
            shift_logits = logits[:-k, :-1, :].contiguous()
            bs = text_tokens["input_ids"].shape[0] - k
        return outputs, logits, labels, shift_labels, shift_logits, bs


    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        
        if k != 1:
            raise AssertionError("Not support Batch Edit")
        
        pmrl_loss = torch.tensor(0.0, device=self.device if hasattr(self, 'device') else multimodal_inputs['input_ids'].device)

        if self.config.using_extra:
            # blip2中，前32个token是image embedding
            # base_features = self.hidden_states[0, :-ans_token_len, :]
            rephrase_samples = []
            if self.config.using_dropout:
                inputs_embeds, attention_mask, targets = self.model.image_encoding(multimodal_inputs[0])
                # img_part = inputs_embeds[:, :-1, :] 
                # txt_part = inputs_embeds[:, -1:, :]
                img_part = inputs_embeds[:, :-ans_token_len, :]
                txt_part = inputs_embeds[:, -ans_token_len:, :]
                # img_part = inputs_embeds[:, :32, :] 
                # txt_part = inputs_embeds[:, 32:, :]
                sample_nums = 2
                noisy_img_sample = []
                for _ in range(sample_nums):
                    if True:
                        # embedding上直接扰动的方法
                        noisy_img_part = F.dropout(img_part, p=0.1, training=True)
                        perturbed_inputs_embeds = torch.cat([noisy_img_part, txt_part], dim=1)
                        # perturbed_inputs_embeds = noisy_img_part
                        self.model.LLM_forward(perturbed_inputs_embeds, attention_mask, targets)
                    else:
                        # simCSE实现方法
                        perturbed_inputs_embeds = copy.deepcopy(inputs_embeds)
                        self.model.LLM_forward(perturbed_inputs_embeds, attention_mask, targets, using_dropout=True)
                    current_perturbed_output = super().get_adapter_layer().new_weight_layer_output
                    rephrase_samples.append(current_perturbed_output)
                outputs, logits, labels, shift_labels, shift_logits, bs = self.mllm_forward(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len, k)
                base_output = super().get_adapter_layer().new_weight_layer_output
                base_output = base_output.reshape(2, -1, base_output.size(-1))[0].detach()
                rephrase_samples.append(base_output)
                # print("base_output shape: ", base_output)
                # print("perturbed_output shape: ", perturbed_output)
            elif self.config.using_LAP:
                if True:
                    # 0. 获取生成数量，默认为1
                    num_rephrase = getattr(self.config, 'num_rephrase', 5)

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
                    epsilon = getattr(self.config, 'lap_epsilon', 1e-3)

                    # 6. 循环生成多个样本
                    for i in range(num_rephrase):
                        # 计算扰动 Delta
                        # 注意：如果需要每个样本不同，通常在这里加入随机噪声，例如：
                        # noise = torch.randn_like(grad_base) * some_scale
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
                else:
                    # 在编辑层的实现方法（untested）
                    outputs, logits, labels, shift_labels, shift_logits, bs = self.mllm_forward(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len, k)
                    
                    a = shift_logits.view(-1, shift_logits.size(-1))
                    b = shift_labels.view(-1)[-ans_token_len:]
                    a = a[-b.size(0):,:]

                    base_hidden_state_input = super().get_adapter_layer().hidden_state_input
                    embeds_for_grad = base_hidden_state_input.detach().clone()
                    embeds_for_grad.requires_grad_(True)
                    
                    loss_fct_inner = torch.nn.CrossEntropyLoss(reduction='sum')
                    J_LM = loss_fct_inner(a, b)

                    # 5. 反向传播求导
                    grad_full = torch.autograd.grad(
                        outputs=J_LM,
                        inputs=embeds_for_grad,
                        retain_graph=False,
                        only_inputs=True,
                        allow_unused=True
                    )[0]

                    # 6. 计算扰动 Delta
                    grad_base = grad_full[:, :32, :]
                    epsilon = getattr(self.config, 'lap_epsilon', 1e-3)
                    
                    grad_norm = torch.norm(grad_base, dim=-1, keepdim=True) + 1e-8
                    delta = (grad_base / grad_norm) * epsilon
                    
                    # 7. 应用扰动并生成样本 (Rephrase Sample 1)
                    noisy_img_part = base_hidden_state_input[:, :32, :] + delta.detach()
                    txt_part = base_hidden_state_input[:, 32:, :]
                    perturbed_inputs_embeds = torch.cat([noisy_img_part, txt_part], dim=1)
                    
                    # 再次前向传播以更新 adapter 状态
                    self.model.LLM_forward(perturbed_inputs_embeds, attention_mask, targets)
                    perturbed_output = super().get_adapter_layer().new_weight_layer_output
                    rephrase_samples.append(perturbed_output)

                # 8. 获取 Base Output (Rephrase Sample 2)
                # 传入原始 embedding 再次前向传播
                self.model.LLM_forward(inputs_embeds, attention_mask, targets)
                base_output = super().get_adapter_layer().new_weight_layer_output
                rephrase_samples.append(base_output.detach())

                # 恢复正常的 outputs 用于函数返回
                outputs, logits, labels, shift_labels, shift_logits, bs = self.mllm_forward(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len, k)
            pmrl_loss = self.pmrl_loss(rephrase_samples, tau_alignment=self.config.pmrl_tau_alignment, tau_regularization=self.config.pmrl_tau_regularization) * self.config.pmrl_scale
        else:
            outputs, logits, labels, shift_labels, shift_logits, bs = self.mllm_forward(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len, k)

        print(f"pmrl_loss: {pmrl_loss.item()}")


        # only cal loss of target text tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        a = shift_logits.view(-1, shift_logits.size(-1))
        b = shift_labels.view(-1)[-ans_token_len:]
        a = a[-b.size(0):,:]
        loss = loss_fct(a, b)
        loss = loss.view(bs, -1)
        label_mask = torch.ones_like(loss, dtype=torch.bool)        
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss + pmrl_loss

            # elif self.config.using_LAP:
            #     # --- 1. 获取基础特征（保持在计算图中） ---

            #     # --- 2. 独立计算 delta（不使用显式 lm_head） ---
            #     # 找到 answer 对应的 logits 部分
                
            #     a = shift_logits.view(-1, shift_logits.size(-1))
            #     b = shift_labels.view(-1)[-ans_token_len:]
            #     a = a[-b.size(0):,:]
            #     # 计算用于探测方向的临时 Loss
            #     # 注意：这个 Loss 必须是在计算图上的
            #     loss_fct_inner = torch.nn.CrossEntropyLoss(reduction='sum')
            #     J_LM = loss_fct_inner(a, b)

            #     # 核心：直接对中间变量 self.hidden_states 求导
            #     # retain_graph=True 是必须的，因为后面真正的 ft_loss 还需要用到这个计算图
            #     grad_full = torch.autograd.grad(
            #         outputs=J_LM,
            #         inputs=self.hidden_states,
            #         retain_graph=True,
            #         only_inputs=True,
            #         allow_unused=True
            #     )[0]

            #     if grad_full is not None:
            #         # 截取对应 answer token 位置的梯度
            #         # grad_base = grad_full[0, -ans_token_len:, :]
            #         grad_base = grad_full[0, :-ans_token_len, :]
                    
            #         # 构造 delta 数值（剥离计算图）
            #         epsilon = getattr(self.config, 'lap_epsilon', 1e-3)
            #         # 使用 .data 确保 delta 是纯数值，不带梯度
            #         delta = (grad_base.data / (torch.norm(grad_base.data) + 1e-8)) * epsilon
            #         # print(delta)
            #     else:
            #         delta = torch.zeros_like(base_features.data)

            #     # --- 3. 构造 PMRL 视图 ---
            #     # 锚点视图：关联计算图，用于传导 pmrl_loss 的梯度回到模型
            #     view_anchor = base_features 
                
            #     # 对抗视图：锚点数值 + 偏移量，并完全 detach
            #     # 它作为 PMRL 的“静态目标”
            #     view_adversarial = (base_features.data + delta).detach()

            #     rephrase_samples = [view_anchor, view_adversarial]
            # else:
            #     raise ValueError("unknown method")
            # 计算 PMRL Loss