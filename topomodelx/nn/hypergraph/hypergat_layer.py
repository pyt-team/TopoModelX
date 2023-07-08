"""HyperGAT layer."""
import torch

from topomodelx.base.conv import Conv


class HyperGATLayer(torch.nn.Module):
    """Implementation of the HyperGAT layer proposed in [DWLLL20].

    References
    ----------
    .. [DWLLL20] Kaize Ding, Jianling Wang, Jundong Li, Dingcheng Li, & Huan Liu. Be more with less:
        Hypergraph attention networks for inductive text classification. In Proceedings of the 2020 Conference
        on Empirical Methods in Natural Language Processing (EMNLP), 2020 (https://aclanthology.org/2020.emnlp-main.399.pdf)

    Parameters
    ----------
    in_features : int
        Dimension of the input features.
    out_features : int
        Dimension of the output features.

    """

    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(HyperGATLayer, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
