import os
import shutil

# Carpetas actuales
image_dir = "dataset/images"
caption_dir = "dataset/captions"

# Nueva raíz para entrenamiento LoRA
output_dir = "dataset_lora"
categoria = "ropa"  # puedes llamarla 'camisetas', 'ropa_deportiva', etc.

output_subdir = os.path.join(output_dir, categoria)
os.makedirs(output_subdir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(image_dir, filename)
        txt_path = os.path.join(caption_dir, f"{base}.txt")

        if os.path.exists(txt_path):
            shutil.copy(img_path, os.path.join(output_subdir, filename))
            shutil.copy(txt_path, os.path.join(output_subdir, f"{base}.txt"))
        else:
            print(f"⚠️ No se encontró descripción para {filename}")

print("✅ Dataset reorganizado para kohya_ss.")
