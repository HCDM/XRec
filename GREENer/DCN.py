import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(
                torch.Tensor(self.layer_num, in_features, 1)
            )
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(
                torch.Tensor(self.layer_num, in_features, in_features)
            )
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs):
        # inputs: (batch_size, units)
        # x_0: (batch_size, units, 1)
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)   # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]                  # W * xi + b
                x_l = x_0 * dot_ + x_l                      # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        # x_l: (batch_size, units)
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **init_seed**: float. Used for initialization of the linear layer.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc

        return deep_input


class DCN(nn.Module):
    """Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
    and DCN-M (parameterization='matrix').

    :param node_embed_size: Node embedding size. User/Item/Sentence should have the same node embedding size.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"`` (or other cuda, if we have)
    :return: A PyTorch model instance.

    """
    def __init__(self, node_embed_size, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=[128, 128], l2_reg_linear=0.00001, l2_reg_cross=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, device='cpu'):
        super(DCN, self).__init__()
        self.node_embed_size = node_embed_size
        self.input_embed_size = self.node_embed_size * 3    # concat user/item/sentence node embeddings
        # after DNN and CrossNet, we concat the 2 feature embeddings
        self.dcn_output_size = self.input_embed_size + dnn_hidden_units[-1]
        # init DNN
        self.dnn = DNN(
            inputs_dim=self.input_embed_size,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            init_std=init_std,
            device=device)
        # init CrossNet
        self.crossnet = CrossNet(
            in_features=self.input_embed_size,
            layer_num=cross_num,
            parameterization=cross_parameterization,
            device=device)
        self.dcn_linear = nn.Linear(
            self.dcn_output_size, 1, bias=False).to(device)
        self.to(device)

    # def forward(self, x, verbose=False):
    def forward(self, x):
        # X: (batch_size, node_embed_size*3)
        deep_out = self.dnn(x)
        cross_out = self.crossnet(x)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)
        logit = self.dcn_linear(stack_out)
        # if verbose:
        #     print("in dcn, x shape: {}".format(x.shape))
        #     print("in dcn, dnn output shape: {}".format(deep_out.shape))
        #     print("in dcn, cross output shape: {}".format(cross_out.shape))
        #     print("in dcn, dnn&cross output shape: {}".format(stack_out.shape))
        #     print("in dcn, logit shape: {}".format(logit.shape))
        # logit = logit.view(-1)
        # y_pred = torch.sigmoid(logit)
        return logit
