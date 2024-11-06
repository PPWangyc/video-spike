import torch

class Linear(torch.nn.Module):
    def __init__(self, config):
        super(Linear, self).__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.output_dim = config.decoder.output_dim // 100

    def forward(self, x):
        x = x.flatten(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, 100, self.output_dim)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer_num = config.layer_num
        hidden_dims = config.hidden_dims
        assert len(hidden_dims) == layer_num, "Encoder hidden_dims must have the same length as layer_num"
        self.layers = torch.nn.Sequential()
        for i in range(layer_num):
            if i == 0:
                self.layers.append(torch.nn.Linear(config.input_dim, hidden_dims[i]))
                self.layers.append(torch.nn.ReLU())
            else:
                self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(torch.nn.ReLU())
        
        self.layers.append(torch.nn.Linear(hidden_dims[-1], config.output_dim))


    def forward(self, x):
        return self.layers(x)
    
class Decoder(torch.nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        layer_num = config.layer_num
        hidden_dims = config.hidden_dims
        assert len(hidden_dims) == layer_num, "Decoder hidden_dims must have the same length as layer_num"
        self.layers = torch.nn.Sequential()
        for i in range(layer_num):
            if i == 0:
                self.layers.append(torch.nn.Linear(config.input_dim, hidden_dims[i]))
                self.layers.append(torch.nn.ReLU())
            else:
                self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(hidden_dims[-1], config.output_dim))

    def forward(self, x):
        return self.layers(x)