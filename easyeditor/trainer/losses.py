import torch
import torch.nn.functional as F


def kl_loc_loss(pre, post, mask=None):
    # Keep logits in their native dtype until bounded chunks are processed.
    # Casting the complete LLaVA-OV vocabulary tensor to fp32 first can
    # allocate >1 GiB while MEND retains the inner computation graph.
    sequence = pre.dim() == 3
    if sequence:
        if pre.shape[-1] <= 1:
            raise NotImplementedError
        if mask is None:
            raise AssertionError("sequence KL locality requires a mask")
        pre_ = pre.contiguous().view(-1, pre.shape[-1])
        post_ = post.contiguous().view(-1, post.shape[-1])
        assert pre_.shape == post_.shape
        mask_ = mask.view(pre_.shape[0])
        denom = mask_.sum()
        if denom == 0:
            raise ValueError("KL locality mask is empty")
        total = torch.zeros((), device=pre.device, dtype=torch.float32)
        chunk_size = 128
        for start in range(0, pre_.shape[0], chunk_size):
            stop = min(start + chunk_size, pre_.shape[0])
            pre_chunk = pre_[start:stop].float()
            post_chunk = post_[start:stop].float()
            kl = (
                pre_chunk.softmax(-1)
                * (pre_chunk.log_softmax(-1) - post_chunk.log_softmax(-1))
            ).sum(-1)
            total = total + (kl * mask_[start:stop].float()).sum()
        return total / denom.float()

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
    pred = pred.to(torch.float32)

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