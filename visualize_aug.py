import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader.data_loaders import LanenetDataLoader

def visualize_batch(loader, num_sequences=2):
    # Obtener un batch
    batch = next(iter(loader))
    images = batch['input_tensor']  # [B, T*3, H, W] o [B, T, 3, H, W] dep. de tu implementación final
    binary_masks = batch['binary_label']
    instance_masks = batch['instance_label']

    print(f"Shape imágenes: {images.shape}")
    print(f"Shape máscaras binarias: {binary_masks.shape}")

    # Asumimos que data_loaders devuelve [B, T*3, H, W] plano, 
    # ajusta si ya lo devuelve reshaped
    if len(images.shape) == 4:
        # Reshape manual para visualizar: [B, T, 3, H, W]
        # Asumiendo T=3 (sequence_length)
        B, C_total, H, W = images.shape
        T = 3 
        images = images.view(B, T, 3, H, W)

    # Visualizar
    for b in range(num_sequences):
        fig, axes = plt.subplots(T, 3, figsize=(15, 8))
        fig.suptitle(f'Secuencia {b} - Augmentations activadas', fontsize=16)
        
        for t in range(T):
            # Imagen original (normalizada o float)
            img = images[b, t].permute(1, 2, 0).cpu().numpy()
            # Si normalizaste, des-normaliza aquí para ver colores reales
            # img = img * std + mean 
            
            # Máscara binaria
            bin_mask = binary_masks[b].cpu().numpy() # [H, W] - Ojo: las máscaras suelen ser 2D por frame?
            # Si tu máscara es [B, H, W] y representa el último frame, o [B, T, H, W]
            # Ajusta según tu dataloader. Asumiré que la máscara corresponde al último frame (t=T-1)
            
            # NOTA: Si tu dataloader devuelve máscaras para TODA la secuencia, úsalas.
            # Si solo devuelve para el frame objetivo (último), solo visualiza ese.
            
            axes[t, 0].imshow(np.clip(img, 0, 1))
            axes[t, 0].set_title(f'Frame t-{T-1-t}')
            axes[t, 0].axis('off')
            
            # Solo mostramos máscara si corresponde a este frame (depende de tu Dataset)
            axes[t, 1].text(0.5, 0.5, "Verificar Máscara \nvs Imagen", ha='center')
            axes[t, 1].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Configura tu loader con augmentations ACTIVADAS
    loader = LanenetDataLoader(
        txt_path="./data/training_data_example/train.txt", # Ajusta rutas
        batch_size=2,
        is_training=True,
        use_augmentation=True # <--- IMPORTANTE
    )
    visualize_batch(loader)