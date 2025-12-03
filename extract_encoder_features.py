# coding: utf-8
"""
Script para extraer características del encoder de LaneNet,
aplicar PCA a 2 dimensiones y visualizar los resultados.
"""
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

from model.lanenet.LaneNet import LaneNet, DEVICE


class EncoderFeatureExtractor(nn.Module):
    """Wrapper para extraer características del encoder"""
    def __init__(self, model):
        super(EncoderFeatureExtractor, self).__init__()
        self.encoder = model._encoder
        self.arch = model._arch
        
    def forward(self, x):
        """Extrae características del encoder y aplica global average pooling"""
        if self.arch == 'UNet':
            c1, c2, c3, c4, c5 = self.encoder(x)
            # Usar la última capa (c5) para extraer características
            features = F.adaptive_avg_pool2d(c5, (1, 1))
            features = features.view(features.size(0), -1)
        elif self.arch == 'ENet':
            c = self.encoder(x)
            features = F.adaptive_avg_pool2d(c, (1, 1))
            features = features.view(features.size(0), -1)
        elif self.arch == 'DeepLabv3+':
            c1, c2 = self.encoder(x)
            # Concatenar ambas características
            feat1 = F.adaptive_avg_pool2d(c1, (1, 1))
            feat2 = F.adaptive_avg_pool2d(c2, (1, 1))
            features = torch.cat([feat1, feat2], dim=1)
            features = features.view(features.size(0), -1)
        else:
            raise ValueError(f"Arquitectura no soportada: {self.arch}")
        return features


def load_image(img_path, transform):
    """Carga y transforma una imagen"""
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img


def extract_features_from_directory(data_dir, model_path, arch='ENet', 
                                    resize_height=512, resize_width=512):
    """
    Extrae características del encoder para todas las imágenes en subdirectorios
    
    Args:
        data_dir: Directorio con subdirectorios de imágenes
        model_path: Ruta al checkpoint del modelo
        arch: Arquitectura del modelo ('ENet', 'UNet', 'DeepLabv3+')
        resize_height: Altura para redimensionar
        resize_width: Ancho para redimensionar
    
    Returns:
        features: Array numpy con características (n_imágenes, n_features)
        labels: Lista con nombres de subdirectorios para cada imagen
        image_paths: Lista con rutas de las imágenes procesadas
    """
    # Setup transforms
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar modelo
    print(f"Cargando modelo {arch} desde {model_path}...")
    model = LaneNet(arch=arch, freeze_encoder=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    # Crear extractor de características
    feature_extractor = EncoderFeatureExtractor(model)
    feature_extractor.eval()
    feature_extractor.to(DEVICE)
    
    # Buscar subdirectorios
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) != 2:
        raise ValueError(f"Se esperaban 2 subdirectorios, se encontraron {len(subdirs)}: {subdirs}")
    
    print(f"Encontrados subdirectorios: {subdirs}")
    
    all_features = []
    all_labels = []
    all_image_paths = []
    
    # Extensiones de imagen soportadas
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    # Procesar cada subdirectorio
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(data_dir, subdir)
        print(f"\nProcesando subdirectorio: {subdir}")
        
        # Buscar todas las imágenes
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(subdir_path, ext)))
        
        image_paths = sorted(image_paths)
        print(f"  Encontradas {len(image_paths)} imágenes")
        
        if len(image_paths) == 0:
            print(f"  Advertencia: No se encontraron imágenes en {subdir_path}")
            continue
        
        # Procesar cada imagen
        for img_path in image_paths:
            try:
                # Cargar imagen
                img_tensor = load_image(img_path, data_transform)
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                
                # Extraer características
                with torch.no_grad():
                    features = feature_extractor(img_tensor)
                    features_np = features.cpu().numpy().flatten()
                
                all_features.append(features_np)
                all_labels.append(subdir)
                all_image_paths.append(img_path)
                
            except Exception as e:
                print(f"  Error procesando {img_path}: {e}")
                continue
    
    if len(all_features) == 0:
        raise ValueError("No se pudieron procesar imágenes")
    
    print(f"\nTotal de imágenes procesadas: {len(all_features)}")
    
    # Convertir a array numpy
    features_array = np.array(all_features)
    
    return features_array, all_labels, all_image_paths


def apply_pca_and_plot(features, labels, output_path='encoder_pca_visualization.png'):
    """
    Aplica PCA a 2 dimensiones y visualiza los resultados
    
    Args:
        features: Array numpy con características (n_imágenes, n_features)
        labels: Lista con etiquetas (nombres de subdirectorios)
        output_path: Ruta para guardar el gráfico
    """
    print(f"\nAplicando PCA a {features.shape[1]} dimensiones -> 2 dimensiones...")
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    print(f"Varianza explicada por cada componente:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    
    # Obtener etiquetas únicas
    unique_labels = sorted(list(set(labels)))
    colors = ['blue', 'red']  # Colores para cada subdirectorio
    
    if len(unique_labels) > 2:
        # Si hay más de 2, usar colormap
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab10')
        colors = [color_map(i) for i in range(len(unique_labels))]
    
    # Crear gráfico
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[i], label=label, alpha=0.6, s=50)
    
    plt.xlabel(f'Primera Componente Principal (PC1) - {pca.explained_variance_ratio_[0]*100:.2f}% varianza')
    plt.ylabel(f'Segunda Componente Principal (PC2) - {pca.explained_variance_ratio_[1]*100:.2f}% varianza')
    plt.title('Visualización PCA de Características del Encoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado en: {output_path}")
    
    # Mostrar gráfico (solo si hay display disponible)
    try:
        plt.show()
    except:
        print("No se pudo mostrar el gráfico (posiblemente sin display). El archivo fue guardado correctamente.")
    
    return features_2d, pca


def save_features(features, labels, image_paths, output_path='encoder_features.npz'):
    """Guarda las características extraídas"""
    np.savez(output_path, 
             features=features, 
             labels=np.array(labels), 
             image_paths=np.array(image_paths))
    print(f"Características guardadas en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extraer características del encoder y visualizar con PCA')
    parser.add_argument('--data_dir', type=str, default='data/test_encoders',
                       help='Directorio con subdirectorios de imágenes')
    parser.add_argument('--model', type=str, default='log/best_model.pth',
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--arch', type=str, default='ENet', 
                       choices=['ENet', 'UNet', 'DeepLabv3+'],
                       help='Arquitectura del modelo')
    parser.add_argument('--height', type=int, default=512,
                       help='Altura para redimensionar imágenes')
    parser.add_argument('--width', type=int, default=512,
                       help='Ancho para redimensionar imágenes')
    parser.add_argument('--output_plot', type=str, default='encoder_pca_visualization.png',
                       help='Ruta para guardar el gráfico PCA')
    parser.add_argument('--output_features', type=str, default='encoder_features.npz',
                       help='Ruta para guardar las características extraídas')
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio
    if not os.path.exists(args.data_dir):
        print(f"Error: El directorio {args.data_dir} no existe.")
        print("Por favor, crea el directorio con dos subdirectorios de imágenes.")
        return
    
    # Extraer características
    features, labels, image_paths = extract_features_from_directory(
        args.data_dir, 
        args.model, 
        arch=args.arch,
        resize_height=args.height,
        resize_width=args.width
    )
    
    # Guardar características
    save_features(features, labels, image_paths, args.output_features)
    
    # Aplicar PCA y visualizar
    features_2d, pca = apply_pca_and_plot(features, labels, args.output_plot)
    
    print("\n¡Proceso completado!")


if __name__ == '__main__':
    main()

