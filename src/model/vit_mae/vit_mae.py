from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
import torch.nn as nn
import torch
import wandb

class ContrastViTMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = ViTMAEConfig(**config)
        self.vit_mae = ViTMAE(self.config)
        self.proj = nn.Linear(self.config.hidden_size, self.config.embed_size)
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        cls_token, recon_loss = self.vit_mae(pixel_values=x)
        z = self.proj(cls_token)
        # normalize projection
        z = z / z.norm(dim=-1, keepdim=True)
        z = cls_token
        return {
            'z': z,
            'recon_loss': recon_loss,
            'temp': self.temperature
        }

class ContrastViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = ViTMAEConfig(**config)
        # set mask_ratio to 0 to disable masking
        self.config.mask_ratio = 0
        self.vit = ViTMAEModel(self.config)
        self.proj = nn.Linear(self.config.hidden_size, self.config.embed_size)
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        cls_token = self.vit(pixel_values=x).last_hidden_state[:, 0]
        z = self.proj(cls_token)
        # normalize projection
        z = z / z.norm(dim=-1, keepdim=True)
        return {
            'z': z,
            'temp': self.temperature
        }

class ViTMAE(ViTMAEForPreTraining):
    def forward(
        self,
        pixel_values,
        noise=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interpolate_pos_encoding=False
    ):
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        latent = outputs.last_hidden_state
        # extract cls latent
        cls_latent = latent[:, 0] # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(pixel_values, logits, mask)
        
        return cls_latent, loss
