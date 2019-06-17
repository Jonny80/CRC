from torch.autograd import Variable


VOID_LABEL = 255
N_CLASSES = 11


def crossentropyloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, Variable(label))
    target = label.view(-1)[mask]
    C = logits.size(1)
    logits = logits.permute(0, 2, 3, 1)  # B, H, W, C
    logits = logits.contiguous().view(-1, C)
    mask2d = mask.unsqueeze(1).expand(mask.size(0), C).contiguous().view(-1)
    logits = logits[mask2d].view(-1, C)
    loss = F.cross_entropy(logits, Variable(target))
    return loss