import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class CRNN(nn.Module):
    def __init__(
        self,
        img_h: int,
        num_chars: int,
        device: torch.device,
        dims: int = 1024,
    ):
        super().__init__()
        self.height = img_h

        self.device = device

        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        cnn_dims = self._cnn_dims()
        print(cnn_dims)
        self.linear = nn.Linear(cnn_dims, dims)
        self.drop = nn.Dropout(0.5)  # ??

        self.rnn = nn.GRU(
            dims, dims // 2, bidirectional=True, num_layers=1, batch_first=True
        )

        self.projection = nn.Linear(dims, num_chars)

        self.cross_entropy = nn.CrossEntropyLoss().to(self.device)

    def load(self, model: str):
        try:
            checkpoint = torch.load(model, map_location=self.device)
            self.load_state_dict(checkpoint["model_state"])
        except Exception as e:
            print(f"Error loading pretrained model: {e}")

    def _cnn_dims(self):
        # Width can be set to any resonable value
        width, height = 1024, self.height
        channel = 3  # Assuming RGB
        dummy_input = torch.zeros(1, channel, height, width)

        return self._encode_part(dummy_input).shape[-1]

    def _encode_part(self, x: Tensor) -> Tensor:
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)

        x = self.cnn.layer2(x)

        # ???
        x = x.permute(0, 3, 1, 2)
        return x.view(x.size(0), x.size(1), -1)

    def encode(self, x: Tensor) -> Tensor:
        x = self._encode_part(x)
        x = self.linear(x)
        x = nn.functional.relu(x)  # ??
        return self.drop(x)

    def forward(
        self,
        images: Tensor,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        features = self.encode(images)
        hiddens, _ = self.rnn(features)

        x = self.projection(hiddens)

        x = x.permute(1, 0, 2)

        if targets is not None:
            # loss = self.nll_loss(x, targets, x.shape[1])
            loss = self.ctc_loss(x, targets, input_lengths, target_lengths)
            return x, loss
        return x, None

    def nll_loss(self, x, targets, seq_len: int):
        targets = self.pad_targets(targets, seq_len)
        scalar = 20  # arbitrary multiplier, TODO: test different values
        return (
            self.cross_entropy(
                x.view(-1, x.shape[-1]), targets.contiguous().view(-1)
            )
            * scalar
        )

    @staticmethod
    def ctc_loss(x, targets, input_len, target_len):
        size = (x.size(1),)
        log_probs = nn.functional.log_softmax(x, 2)

        if input_len is None:
            input_len = torch.full(size, log_probs.size(0), dtype=torch.int32)
        if target_len is None:
            target_len = torch.full(size, targets.size(1), dtype=torch.int32)

        loss = nn.CTCLoss(blank=0)(log_probs, targets, input_len, target_len)
        return loss

    def pad_targets(self, targets: Tensor, seq_len: int):  # ??
        padding = (0, seq_len - targets.shape[1])
        return nn.functional.pad(targets, padding, "constant", 0)
