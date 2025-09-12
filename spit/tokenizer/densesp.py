import torch
import torch.nn as nn

from .tokenizer import diffmap_from_seg, concatenate_diffmap, superpixel_tokenizer

class DenseSPEdgeEmbedder(nn.Module):

    '''Embeds superpixel segmentation maps into dense edge maps.

    This allows for embedding superpixel information into image representations
    without losing the original image resolution.

    Example:
    ```
        >>> import torch
        >>> from spit.tokenizer.densesp import DenseSPEdgeEmbedder
        >>> img = ... # (1, 3, 224, 224) input image
        >>> tokenizer = DenseSPEdgeEmbedder()
        >>> out = tokenizer(img)
        >>> out.shape
        torch.Size([1, 5, 224, 224])
        # Out can then be fed to a 5-channel Conv2d layer
    ```
    '''

    def __init__(self, maxlvl:int=4, drop_delta:bool=False, bbox_reg:bool=True, learn_vrange:bool=True, return_seg:bool=False):
        super().__init__()
        self._lgrad = 27.8
        self._lcol = 10.
        self._maxlvl = 4
        self.drop_delta = drop_delta
        self.bbox_reg = bbox_reg
        self.return_seg = return_seg
        if learn_vrange:
            self.vmin = nn.Parameter(torch.tensor(-torch.pi/2))
            self.vmax = nn.Parameter(torch.tensor(torch.pi/2))
        else:
            self.register_buffer('vmin', torch.tensor(-torch.pi/2))
            self.register_buffer('vmax', torch.tensor(torch.pi/2))

    def forward(self, img:torch.Tensor):
        _, seg, _ = superpixel_tokenizer(
            img, self._lgrad, self._lcol, self.drop_delta, self.bbox_reg, 
            False, self._maxlvl,
        )
        img = concatenate_diffmap(img, seg, vmin=self.vmin, vmax=self.vmax) # type: ignore
        if self.return_seg:
            return img, seg
        return img
