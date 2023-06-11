from torch import nn 
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self,input_shape, hidden_layer_1, hidden_layer_2, num_classes):
    super().__init__()
    self.linear1 = nn.Linear(input_shape,hidden_layer_1)
    self.linear2 = nn.Linear(hidden_layer_1,hidden_layer_2)
    self.linear3 = nn.Linear(hidden_layer_2,num_classes)

  def formard(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)  # this is the score
    return x