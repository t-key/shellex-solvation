import torch
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(7168,256),torch.nn.SiLU(),torch.nn.Dropout(.1),torch.nn.Linear(256,128),torch.nn.SiLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,3))
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)
