import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from safetensors.torch import load_file
from PIL import Image

# Cargar modelo base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe.to("cuda", dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# ⚠️ Reemplazar los módulos de atención por LoRAAttnProcessor con hidden_size fijo
for name, module in pipe.unet.named_modules():
    if hasattr(module, "set_attn_processor"):
        try:
            module.set_attn_processor(LoRAAttnProcessor(hidden_size=768))
        except Exception as e:
            print(f"Saltando {name}: {e}")

# ✅ Cargar pesos del LoRA entrenado
lora_weights = load_file("D:/Entrenamiento/output/lora_ropa_deportiva.safetensors")
pipe.unet.load_state_dict(lora_weights, strict=False)

# 🔥 Prompt de prueba
prompt = "<lora:lora_ropa_deportiva:1> Crea la imagen de un carro blanco"

# Generar imagen
image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("imagen_lora_final.png")
print("✅ Imagen generada y guardada como imagen_lora_final.png")
