# -*- coding: utf-8 -*-
import torch.nn as nn


class ModuleUnit(nn.Module):
    def __init__(self, input_size, output_size, bias=True, hidden_size=None, hidden_layers=None, activation=None, dropout=None):
        super(ModuleUnit, self).__init__()
        fc_ac = {"relu": "nn.ReLU",
                 "leaky_relu": "nn.LeakyReLU",
                 "tanh": "nn.Tanh",
                 "sigmoid": "nn.Sigmoid"
                 }

        model_list = []
        if hidden_layers is None or hidden_layers == 0:
            model_list.append(nn.Linear(input_size, output_size, bias=bias))
        else:
            for n in range(hidden_layers):
                if n == 0:
                    model_list.append(nn.Linear(input_size, hidden_size, bias=bias))
                else:
                    model_list.append(nn.Linear(hidden_size, hidden_size, bias=bias))
                m = model_list[-1]

                if activation is not None and activation["fc_name"] != "":
                    if activation["fc_name"] in fc_ac.keys():
                        if activation["fc_name"] in ["sigmoid", "tanh"]:
                            nn.init.xavier_normal_(m.weight.data, gain=1)
                            if m.bias is not None:
                                m.bias.data.zero_()
                        else:
                            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=activation["fc_name"])

                        if "params" in activation.keys() and activation["params"] != "":
                            _add_module = eval(fc_ac[activation["fc_name"]] + "(" + activation["params"] + ")")
                            model_list.append(_add_module)
                        else:
                            _add_module = eval(fc_ac[activation["fc_name"]] + "()")
                            model_list.append(_add_module)
                    else:
                        raise ValueError("Activation function is not available")

                if dropout is not None and dropout != 0:
                    if 0 <= dropout < 1:
                        model_list.append(nn.Dropout(p=dropout))
                    else:
                        raise ValueError("Error: Incorrect dropout rate!")

            model_list.append(nn.Linear(hidden_size, output_size, bias=bias))
        self.model = nn.Sequential(*model_list)

    def forward(self, inputs):
        output = self.model(inputs)
        return output
