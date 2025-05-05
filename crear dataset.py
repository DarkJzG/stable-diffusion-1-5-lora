import os
import pandas as pd

image_dir = "dataset/images"
caption_dir = "dataset/captions"

datos = []

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        caption_file = os.path.join(caption_dir, filename.replace(".png", ".txt"))
        
        if os.path.exists(caption_file):
            with open(caption_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            datos.append({"image_path": image_path, "text": text})
        else:
            print(f"❌ Descripción no encontrada para: {filename}")

# Guardar en CSV
df = pd.DataFrame(datos)
df.to_csv("train.csv", index=False)
print("✅ Archivo train.csv creado correctamente.")
