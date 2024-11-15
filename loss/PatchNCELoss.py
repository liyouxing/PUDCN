from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    """ negative contrastive loss"""

    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)   dim=-1，计算x,y最后一个维度上的向量余弦相似性，其他维度上的shape应该相同，若不同则会进行广播
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q, feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize, 1, -1), feat_k.view(1, batchSize, -1))
        l_neg_curbatch = l_neg_curbatch.view(1, batchSize, -1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


def calculate_NCE_loss(nce_losses, netG_1, netG_2, netF_1, netF_2, src, tgt, nce_layers=None, num_patches=256):
    if nce_layers is None:
        nce_layers = []
    n_layers = len(nce_layers)
    feat_q = netG_1(tgt, nce_layers, encode_only=True)
    feat_k = netG_2(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = netF_1(feat_k, num_patches, None)
    feat_q_pool, _ = netF_2(feat_q, num_patches, sample_ids)
    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, nce_losses, nce_layers):
        loss = crit(f_q, f_k)
        total_nce_loss += loss.mean()
    return total_nce_loss / n_layers
