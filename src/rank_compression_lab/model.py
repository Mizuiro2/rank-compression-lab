import torch
from torch import nn


class Reranker(nn.Module):
    def __init__(self, input_dim: int = 40, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(features))))).squeeze(-1)

    def architecture(self) -> dict:
        return {"input_dim": self.input_dim, "hidden": list(self.hidden)}
