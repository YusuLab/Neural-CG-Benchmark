from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss
import nugets.losses.losses as Losses


from nugets.datasets.datapoint_types import Set_batch, Point_datapoint, LabeledSetBatch
from nugets.models.model import EncoderDecoder

class SetIdentityEncoderDecoder(EncoderDecoder):
    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int|None,
                loss_function:str, absolute_positional_encoding: str | None = None, inject_noise: float = 0.1, *args, **kwargs):
        super().__init__()
        assert loss_function == "directional_width_loss" 
        self.loss_function = getattr(Losses, loss_function)
        self.input_dim = input_dim
    
    def encoder(self, batch: Set_batch):
        return batch, None
    def decode(self, result: Set_batch):
        return result
    def compute_result(self, batch: Set_batch, backbone_result: Set_batch, encoder_info):
        eps_kernel = self.decode(backbone_result)
        return self.loss_function(predicted=eps_kernel, target=batch, in_dim=self.input_dim)