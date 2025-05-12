import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .connector_edit import Qwen2Connector
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock
import lightning as L
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder
from peft import LoraConfig, get_peft_model_state_dict
import prodigyopt
from einops import rearrange, repeat
from diffusers.loaders import LoraLoaderMixin, PeftAdapterMixin
import os
from pathlib import Path
from safetensors.torch import load_file

@dataclass
class Step1XParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool



class Step1XEdit(nn.Module, PeftAdapterMixin):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: Step1XParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.connector = Qwen2Connector()

    @staticmethod
    def timestep_embedding(
        t: Tensor, dim, max_period=10000, time_factor: float = 1000.0
    ):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model

class UnifiedEdit(L.LightningModule):
    def __init__(
        self, 
        params: Step1XParams,
        dit_path: str,
        ae_path: str,
        qwen2vl_model_path: str,
        device: str,
        dtype: torch.bfloat16,
        ae_params: dict[str, any],
        qwen2_params: dict[str, any],  
        lora_path: str = None,
        lora_config: dict = None,
        opt_config: dict = None,
    ):
        super().__init__()
        self.model = Step1XEdit(params)
        self.model = load_state_dict(self.model, dit_path)
        self.ae = AutoEncoder(**ae_params)
        self.ae = load_state_dict(self.ae, ae_path)
        self.mllm = Qwen25VL_7b_Embedder(**qwen2_params)

        self.ae.requires_grad_(False)
        self.ae.eval()
        self.mllm.requires_grad_(False)
        self.mllm.eval()
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.optimizer_config = opt_config
        self.model_dtype = dtype
        self.dit_path = dit_path
        self.ae_path = ae_path
        self.qwen2vl_model_path = qwen2vl_model_path
        # self.learnable_query = nn.Parameter(torch.randn(1, 512, 3584, device=self.device, dtype=dtype))
        # self.learnable_query.requires_grad_(True)
        self.learnable_query = None


    def init_lora(self, lora_path: str, lora_config:dict):
        assert lora_path or lora_config
        if lora_path:
            #TODO
            raise NotImplementedError
        else:
            self.model.add_adapter(LoraConfig(**lora_config))
            lora_layers = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and not "connector" in name:
                    lora_layers.append(param)   
        return lora_layers
    
    def save_weights(self, save_path: str):
        self.model.save_lora_adapter(save_path)
        torch.save(self.model.connector.state_dict(), os.path.join(save_path, "connector.pth"))

    def configure_optimizers(self):
        self.model.requires_grad_(False)
        self.model.connector.requires_grad_(True)

        opt_config = self.optimizer_config
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.model.connector.parameters()))
        self.trainable_params.extend(self.lora_layers)
        # self.trainable_params.append(self.learnable_query)

        for p in self.trainable_params:
            p.requires_grad_(True)

        optimizer = prodigyopt.Prodigy(self.trainable_params, **opt_config['params'])
        # opt_config["params"]["lr"] = 1e-4
        # opt_config["params"]["eps"] = 1e-8
        # opt_config["params"]["betas"] = [0.9, 0.999]
        # optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        return optimizer
        
    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def step(self, batch):
        self.model.train()
        imgs= batch["tgt_imgs"]
        ref_imgs = batch["ref_imgs"]
        prompts = batch["prompts"]
        # ref_img_pil = batch["ref_img_pil"]
        # Embed the reference images

        with torch.no_grad():
            
            ref_img_latents = self.ae.encode(ref_imgs.to(self.device) * 2 -1).to(self.model_dtype)
            x_0 = self.ae.encode(imgs.to(self.device) * 2 -1).to(self.model_dtype)
            bs, _, h, w = x_0.shape
            x_0 = rearrange(x_0, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            ref_img_latents = rearrange(ref_img_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device, dtype=self.model_dtype)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.model_dtype)

            # x_t = rearrange(x_t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

            ref_img_ids = torch.zeros(h // 2, w // 2, 3)
            ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(h // 2)[:, None]
            ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(w // 2)[None, :]
            ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

            if isinstance(prompts, str):
                prompts = [prompts]
            
        llm_embedding, mask = self.mllm(prompts, ref_imgs, self.learnable_query)
        txt_ids = torch.zeros(bs, llm_embedding.shape[1], 3).to(device=x_t.device, dtype=x_t.dtype)
        img = torch.cat([x_t, ref_img_latents.to(device=x_t.device, dtype=x_t.dtype)], dim=-2).to(device=x_t.device, dtype=x_t.dtype)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2).to(device=x_t.device, dtype=x_t.dtype)
        t_vec = torch.full((img.shape[0],), t.item(), dtype=img.dtype, device=img.device)
        txt, vec = self.model.connector(llm_embedding, t_vec, mask)
        preds = self.model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
        )
        pred, _ = preds.chunk(2, dim=1)
        # The pred and x_0's dims are not aligned
        # x_1: torch.Size([1, 16, 64, 64]), pred: torch.Size([1, 2048, 64])
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
                