from torch import nn

class MLP(nn.Module):
  '''
  Multilayer Perceptron

  Inputs:
    input_var: shape=(batch_size, input_size)

  Outputs:
    output_var: shape=(batch_size, layer_sizes[-1])
  '''

  def __init__(
    self,
    d_input,
    layer_sizes,
    act='relu'
  ):
    super().__init__()

    act_fn = dict(
      relu=nn.ReLU,
      sigmoid=nn.Sigmoid,
      tanh=nn.Tanh,
      leakyrelu=nn.LeakyReLU
    )[act]

    layers = []
    i_size = d_input

    for o_size in layer_sizes:
      layers.append(nn.Linear(i_size, o_size))
      layers.append(act_fn())
      i_size = o_size

    self.layers = nn.Sequential(*layers)

  def forward(self, input_var):
    return self.layers(input_var)
