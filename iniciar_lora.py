import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
import gradio as gr
import os

# Verificación de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Cargar modelo base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Aplicar LoRA entrenado
def apply_lora(pipe, lora_path, lora_scale=1.0):
    from diffusers.models.attention_processor import LoRAAttnProcessor

    for name, module in pipe.unet.named_modules():
        if hasattr(module, "set_attn_processor"):
            try:
                hidden_size = module.to_q.in_features
                lora = LoRAAttnProcessor(hidden_size, cross_attention_dim=768, rank=4)
                lora.load_state_dict(load_file(lora_path), strict=False)
                module.set_attn_processor(lora)
            except Exception:
                pass

apply_lora(pipe, "D:/Entrenamiento/output/lora_ropa_deportiva.safetensors")

# Función para generar la imagen
def generar(prenda, color, patron):
    prompt = f"{prenda} deportiva de color {color} con patrón de {patron}, fondo blanco"
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    return image

# Interfaz
gr.Interface(
    fn=generar,
    inputs=[
        gr.Dropdown(["camiseta", "chaqueta", "conjunto exterior"], label="Tipo de prenda"),
        gr.Dropdown(["rojo", "azul", "blanco", "negro", "amarillo"], label="Color principal"),
        gr.Dropdown(["rayas", "puntos", "liso", "cuadros"], label="Patrón")
    ],
    outputs=gr.Image(type="pil"),
    title="Generador de Ropa Deportiva con LoRA"
).launch()
