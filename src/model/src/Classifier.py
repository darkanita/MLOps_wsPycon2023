from torch import nn 
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self,D_in, H1, H2, D_out):
    super().__init__()
    self.linear1 = nn.Linear(D_in,H1)
    self.linear2 = nn.Linear(H1,H2)
    self.linear3 = nn.Linear(H2,D_out)

  def formard(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)  # this is the score
    return x