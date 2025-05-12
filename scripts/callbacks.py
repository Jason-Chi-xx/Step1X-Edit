import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os
import base64
try:
    import wandb
except ImportError:
    wandb = None

from inference import ImageGenerator

class TestImageGenerator(ImageGenerator):
    def __init__(self, dit, ae, llm_encoder, device):
        self.device = torch.device(device)
        self.dit = dit
        self.ae = ae
        self.llm_encoder = llm_encoder
        self.dit.eval()

class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.has_checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if not self.has_checked and trainer.current_epoch == 0 and trainer.global_step == 0:
            # 执行你的检查逻辑
            imgs= batch["tgt_imgs"]
            ref_imgs = batch["ref_imgs"]
            prompts = batch["prompts"]
            # 打印 prompt
            print("=== Sanity Check ===")
            print("Prompt:", prompts)
            
            # 保存图像
            for name, img in [("tgt", imgs), ("ref", ref_imgs)]:
                img_clamped = img.clone().detach().cpu()
                # img_clamped = (img_clamped + 1.0) / 2.0  # 如果是 [-1, 1] 归一化的图像
                img_clamped = torch.clamp(img_clamped, 0, 1)
                # 转换为PIL图像
                img_pil = Image.fromarray((img_clamped[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                if not os.path.exists(os.path.join(self.save_path, "check")):
                    os.makedirs(os.path.join(self.save_path, "check"))
                save_path = os.path.join(self.save_path, "check", f"{name}_img.png")
                img_pil.save(save_path)
                print(f"已保存 {name}_img 到 {save_path}")

            print("====================")
            self.has_checked = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_weights(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
    ):
        # TODO: change this two variables to parameters
        # condition_size = trainer.training_config["dataset"]["condition_size"]
        target_size = trainer.training_config["dataset"]["target_size"]
        # position_scale = trainer.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []
        image_generator = TestImageGenerator(
            dit=pl_module.model,
            ae=pl_module.ae,
            llm_encoder=pl_module.mllm,
            device=pl_module.device,
        )
        test_list = [
            (
                Image.open("assets/cartoon_boy.png").convert("RGB"),
                "close one eye in a wink and slightly open the mouth to form a small smile.",
            ),
            (
                Image.open("assets/aniya.png").convert("RGB"),
                "Change the character's expression by closing one eye in a wink, while the mouth forms a small pout, conveying a shy or bashful emotion. Maintain the character's overall appearance and pose.",
            ),
            (
                Image.open("assets/luffy_3.jpg").convert("RGB"),
                "Shift the expression to angry, narrow the eyes slightly, lower the eyebrows, and slightly open the mouth.",
            ),
            (
                Image.open("assets/saitama_2.jpg").convert("RGB"),
                "Shift the expression to happy, narrow the eyes slightly and open the mouth gently.",
            ),
        ]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, (image_pils, prompt, *others) in enumerate(test_list):
            res = image_generator.generate_image(
                prompt,
                negative_prompt="",
                ref_images=image_pils,
                num_samples=1,
                num_steps=28,
                cfg_guidance=6.0,
                seed=1234,
                show_progress=True,
                size_level=target_size,
                # learnable_query=pl_module.learnable_query,
            )
            res[0].save(
                os.path.join(save_path, f"{file_name}_{i}.jpg")
            )
