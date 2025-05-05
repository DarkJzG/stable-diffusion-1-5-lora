import os
import shutil

carpetas = ["camisetas", "chaquetas", "conjunto exterior"]
ruta_origen = "./prendas"
ruta_destino_img = "./dataset/images"
ruta_destino_txt = "./dataset/captions"

os.makedirs(ruta_destino_img, exist_ok=True)
os.makedirs(ruta_destino_txt, exist_ok=True)

for carpeta in carpetas:
    carpeta_path = os.path.join(ruta_origen, carpeta)
    for archivo in os.listdir(carpeta_path):
        ruta_archivo = os.path.join(carpeta_path, archivo)
        if archivo.endswith(".png"):
            shutil.copy(ruta_archivo, os.path.join(ruta_destino_img, archivo))
        elif archivo.endswith(".txt"):
            shutil.copy(ruta_archivo, os.path.join(ruta_destino_txt, archivo))

print("✅ Dataset organizado en carpetas 'images' y 'captions'.")
