from transformers import VideoMAEModel, AutoImageProcessor
import torch.nn as nn
import torch
class VideoMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.video_mae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.preprocessor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        # uniform sample 16 frames from the video
        indicies = torch.linspace(0, 1, 16)
        self.indicies = (indicies * 119).long()
        self._freeze_params()
        self.encoder = nn.Linear(1568 * 768, config.encoder.output_dim)
        self.decoder = nn.Linear(config.encoder.output_dim, config.decoder.output_dim)

    def forward(self, inputs):
        with torch.no_grad():
            inputs = inputs[:, self.indicies]
            inputs = inputs.squeeze(2).unsqueeze(-1)
            inputs = inputs.repeat(1, 1, 1, 1, 3)
            _batch_size = inputs.shape[0]
            inputs = inputs.reshape(_batch_size*len(self.indicies), inputs.shape[2], inputs.shape[3], inputs.shape[4])
            inputs = self.preprocessor(list(inputs), return_tensors="pt")
            _temp = inputs['pixel_values'].cuda().squeeze(0).clone()
            inputs['pixel_values'] = _temp.reshape(_batch_size, len(self.indicies), _temp.shape[1], _temp.shape[2], _temp.shape[3])
            outputs = self.video_mae(**inputs)
            outputs = outputs.last_hidden_state
            outputs = outputs.flatten(1)
        outputs = self.encoder(outputs)
        outputs = self.decoder(outputs)
        outputs = outputs.reshape(_batch_size, 100, -1)
        return outputs

    def _freeze_params(self):
        for param in self.video_mae.parameters():
            param.requires_grad = False