import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Cargar el modelo base con float32
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
)

# Enviar todo el modelo a CUDA con float32 (esto evita errores con VAE)
pipe.to("cuda", dtype=torch.float32)

# Opcional: cambiar el scheduler por uno más moderno
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Prompt de prueba
prompt = "una camiseta deportiva roja con rayas negras sobre fondo blanco"

# Generar la imagen
image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

# Guardar el resultado
image.save("imagen_base_float32.png")

print("✅ Imagen generada y guardada como imagen_base_float32.png")
