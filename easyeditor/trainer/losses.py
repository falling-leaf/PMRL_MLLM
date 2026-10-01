import torch
import torch.nn.functional as F


def hf_target_loss(logits, labels, attention_mask=None, reduction="mean", chunk_size=128):
    """Compute causal CE only for supervised HF target tokens.

    Multimodal batches contain long image-prefix sequences but usually only a
    few answer tokens.  Selecting the valid shifted rows before softmax avoids
    materialising a full-vocabulary fp32 loss tensor while keeping the same
    ``logits[..., :-1]`` / ``labels[..., 1:]`` convention as Transformers.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"HF target loss expects logits [B,L,V] and labels [B,L], "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"HF logits/labels sequence shapes differ: {tuple(logits.shape[:2])} "
            f"vs {tuple(labels.shape)}"
        )
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError(
                f"HF attention mask shape {tuple(attention_mask.shape)} does not "
                f"match labels {tuple(labels.shape)}"
            )
        valid = labels[:, 1:].ne(-100) & attention_mask[:, 1:].bool()
    else:
        valid = labels[:, 1:].ne(-100)
    if not valid.any():
        raise RuntimeError("HF batch has no supervised target tokens")

    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    valid = valid.reshape(-1)
    selected_logits = shift_logits[valid]
    selected_labels = shift_labels[valid]
    if reduction == "none":
        pieces = []
        for start in range(0, selected_logits.shape[0], chunk_size):
            stop = min(start + chunk_size, selected_logits.shape[0])
            pieces.append(F.cross_entropy(
                selected_logits[start:stop], selected_labels[start:stop], reduction="none"
            ))
        return torch.cat(pieces)
    # A single selected-row CE is already bounded by the number of answer
    # tokens; chunking here additionally bounds temporary fp32 softmax memory.
    if reduction == "sum":
        total = selected_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, selected_logits.shape[0], chunk_size):
            stop = min(start + chunk_size, selected_logits.shape[0])
            total = total + F.cross_entropy(
                selected_logits[start:stop], selected_labels[start:stop], reduction="sum"
            ).float()
        return total
    if reduction != "mean":
        raise ValueError(f"Unsupported HF target-loss reduction: {reduction}")
    return hf_target_loss(
        logits, labels, attention_mask=attention_mask, reduction="sum", chunk_size=chunk_size
    ) / valid.sum().to(torch.float32)


class _ChunkedSequenceKLLoss(torch.autograd.Function):
    """Sequence KL whose forward does not retain per-chunk softmax buffers.

    The locality logits are very large (LLaVA-OV has a 152k vocabulary).  A
    normal Python loop builds one autograd subgraph per row chunk and keeps all
    of their log-softmax intermediates alive until the outer MEND backward.
    The custom backward recomputes the closed-form gradient one chunk at a
    time, retaining only the input logits and one bounded temporary.
    """

    @staticmethod
    def forward(ctx, pre, post, mask, chunk_size):
        if pre.shape != post.shape:
            raise ValueError(f"KL logits differ: {pre.shape} vs {post.shape}")
        if mask.ndim != 1 or mask.shape[0] != pre.shape[0]:
            raise ValueError("KL mask must be flattened to the sequence rows")
        if chunk_size < 1:
            raise ValueError("KL chunk_size must be positive")
        denom = mask.sum()
        if int(denom.item()) == 0:
            raise ValueError("KL locality mask is empty")
        # Saving post is required for autograd's input identity, but no
        # intermediate softmax/log-softmax buffers from the forward are saved.
        ctx.save_for_backward(pre.detach(), post.detach(), mask)
        ctx.chunk_size = int(chunk_size)
        ctx.denom = denom
        total = torch.zeros((), device=post.device, dtype=torch.float32)
        for start in range(0, pre.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, pre.shape[0])
            pre_chunk = pre[start:stop].float()
            post_chunk = post[start:stop].float()
            kl = pre_chunk.softmax(-1) * (
                pre_chunk.log_softmax(-1) - post_chunk.log_softmax(-1)
            )
            total = total + (kl.sum(-1) * mask[start:stop].float()).sum()
        return total / denom.float()

    @staticmethod
    def backward(ctx, grad_output):
        pre, post, mask = ctx.saved_tensors
        # The gradient is consumed by autograd only with respect to post.  Form
        # one bounded chunk at a time; accumulate into a flat CPU buffer so the
        # GPU never needs a second full-vocabulary gradient allocation.
        grad_post = torch.empty(post.shape, device="cpu", dtype=post.dtype, pin_memory=False)
        denom = ctx.denom.to(torch.float32)
        with torch.no_grad():
            for start in range(0, post.shape[0], ctx.chunk_size):
                stop = min(start + ctx.chunk_size, post.shape[0])
                pre_chunk = pre[start:stop].float()
                post_chunk = post[start:stop].float()
                chunk_grad = post_chunk.softmax(-1) - pre_chunk.softmax(-1)
                chunk_grad.mul_(mask[start:stop].float().unsqueeze(-1))
                chunk_grad.div_(denom)
                grad_post[start:stop].copy_(chunk_grad.to("cpu", dtype=post.dtype))
                del pre_chunk, post_chunk, chunk_grad
        # Returning a CPU tensor is not accepted for a CUDA input; transfer in
        # bounded row chunks to avoid materialising the whole GPU gradient.
        result = torch.empty_like(post)
        for start in range(0, post.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, post.shape[0])
            result[start:stop].copy_(grad_post[start:stop].to(post.device))
        del grad_post
        return None, result * grad_output.to(post.dtype), None, None


def kl_loc_loss(pre, post, mask=None, chunk_size=32):
    # Keep the sequence KL's differentiable intermediates bounded.  The custom
    # autograd path is mathematically identical to the chunked reference but
    # does not retain one log-softmax graph per chunk until meta-backward.
    sequence = pre.dim() in (2, 3)
    if sequence:
        if pre.shape[-1] <= 1:
            raise NotImplementedError
        if mask is None:
            raise AssertionError("sequence KL locality requires a mask")
        pre_ = pre.contiguous().view(-1, pre.shape[-1])
        post_ = post.contiguous().view(-1, post.shape[-1])
        mask_ = mask.reshape(-1).bool()
        if mask_.shape[0] != pre_.shape[0]:
            raise ValueError("KL mask does not match flattened sequence rows")
        return _ChunkedSequenceKLLoss.apply(pre_, post_, mask_, int(chunk_size))

    pre = pre.to(torch.float32)
    post = post.to(torch.float32)
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]
    if pre_.shape[-1] == 1:
        return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
            (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
        ).mean()
    raise NotImplementedError


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def multiclass_log_probs(config, pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    # `pred` is read-only below. Cloning the complete [batch, sequence,
    # vocabulary] tensor can cost multiple GiB for LLaVA-OV and OOM during
    # validation while a MEND checkpoint graph is resident.
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs or "personality" in kwargs or "multimodal" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    # Compute only the target-token log probabilities in bounded chunks.  A
    # full-vocabulary fp32 log_softmax is prohibitively large for LLaVA-OV
    # while MEND retains the inner graph.
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_targ = targ.reshape(-1)
    flat_selected = []
    chunk_size = 128
    for start in range(0, flat_pred.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_pred.shape[0])
        flat_selected.append(
            flat_pred[start:stop].float().log_softmax(-1).gather(
                -1, flat_targ[start:stop].unsqueeze(-1)
            ).squeeze(-1)
        )
    unmasked_log_probs = torch.cat(flat_selected).view_as(targ)
    
    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()

        if 't5' in config.model_class.lower():
            end_mask = targ != 1
            correct = correct & end_mask
            num_non_padding = (mask & end_mask).sum().float().item()
        acc = correct.sum() / num_non_padding
    
    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
        
        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }


def masked_log_probs(config, pred, targ, shift=False, exact_match=False, **kwargs):
    # Keep the large HF vocabulary tensor in native bf16; multiclass_log_probs
    # casts only bounded target rows when it computes log-softmax.
    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ, shift=shift, exact_match=exact_match, **kwargs)



def es(pre_logits, post_logits, targ, same_per_mask, q_mask, NULL_TOKEN=0):
    with torch.no_grad():
        
        mask = targ != -100
        targ[~mask] = NULL_TOKEN 
        
        pos_mask = same_per_mask.unsqueeze(-1) * q_mask
        neg_mask = ~same_per_mask.unsqueeze(-1) * q_mask
        
        # Compute log likelihoods of pos/neg samples

        pre_edit_token_log_probs = pre_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
        post_edit_token_log_probs = post_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

        mean_pos_pre = masked_mean(pre_edit_token_log_probs, pos_mask)
        mean_pos_post = masked_mean(post_edit_token_log_probs, pos_mask)
        mean_neg_post = masked_mean(post_edit_token_log_probs, neg_mask)

        z_per = (mean_pos_post - mean_neg_post).sigmoid()
        z_topic_raw = (mean_pos_post - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_post,
            "wrong_probs": mean_neg_post,
        }