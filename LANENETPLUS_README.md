# LaneNetPlus - Enhanced Lane Detection

Este documento describe las mejoras implementadas en LaneNetPlus, una versión extendida de LaneNet con capacidades adicionales.

## Características Principales

### 1. Multi-Task Learning (Lane + Drivable Area)
- Predicción simultánea de máscaras de carriles y área transitable
- Dos decoders separados: `lane_head` y `drivable_head`
- Función de pérdida combinada: `loss = lane_loss + λ * drivable_loss` (λ=0.5 por defecto)

### 2. Homografía Dinámica Offline
- Rectificación de perspectiva usando detección de bordes y RANSAC
- Preprocesamiento de datasets completos
- Opción de activar/desactivar durante entrenamiento e inferencia

### 3. Self-Attention Ligera
- Bloques de atención multi-cabeza en el encoder
- Mejora la capacidad del modelo para capturar dependencias espaciales
- Opcional mediante flag `--use_attention`

## Instalación y Requisitos

Las dependencias son las mismas que el proyecto base. Asegúrate de tener:
- PyTorch
- OpenCV
- NumPy
- PIL/Pillow
- scikit-learn (para PCA en extract_encoder_features.py)

## Uso

### Entrenamiento

#### Entrenamiento básico con LaneNetPlus (multi-task)
```bash
python train.py \
    --dataset data/training_data_example \
    --use_lanenet_plus \
    --use_multitask \
    --drivable_dir data/training_data_example/gt_image_drivable \
    --model_type ENet \
    --epochs 50 \
    --bs 4 \
    --lr 0.0001 \
    --save log/lanenet_plus
```

#### Con atención y rectificación
```bash
python train.py \
    --dataset data/training_data_example \
    --use_lanenet_plus \
    --use_attention \
    --use_multitask \
    --use_rectification \
    --drivable_dir data/training_data_example/gt_image_drivable \
    --model_type ENet \
    --epochs 50 \
    --bs 4 \
    --save log/lanenet_plus_full \
    --save_visualizations
```

### Preprocesamiento con Homografía

#### Rectificar un dataset completo
```bash
python preprocess/rectify_dataset.py \
    --input data/raw_dataset \
    --output data/rectified_dataset
```

Esto procesará todas las imágenes y máscaras en el formato TuSimple.

### Inferencia

#### Inferencia básica
```bash
python test.py \
    --img data/test_image.jpg \
    --model log/lanenet_plus/best_model.pth \
    --use_lanenet_plus \
    --use_multitask \
    --model_type ENet
```

#### Con rectificación
```bash
python test.py \
    --img data/test_image.jpg \
    --model log/lanenet_plus/best_model.pth \
    --use_lanenet_plus \
    --use_multitask \
    --use_rectification \
    --model_type ENet
```

## Estructura de Datos

### Formato de Dataset para Multi-Task

El dataloader espera el siguiente formato:

```
dataset/
├── image/
│   ├── 0000.png
│   └── ...
├── gt_image_binary/
│   ├── 0000.png
│   └── ...
├── gt_image_instance/
│   ├── 0000.png
│   └── ...
├── gt_image_drivable/  # Opcional, para multi-task
│   ├── 0000.png
│   └── ...
├── train.txt
└── val.txt
```

**Nota**: Si las máscaras de drivable area no existen, el dataloader generará máscaras vacías (negras) como placeholders hasta que las agregues.

### Formato de train.txt / val.txt

```
/path/to/image/0000.png /path/to/gt_image_binary/0000.png /path/to/gt_image_instance/0000.png
/path/to/image/0001.png /path/to/gt_image_binary/0001.png /path/to/gt_image_instance/0001.png
...
```

## Parámetros Principales

### Entrenamiento

- `--use_lanenet_plus`: Activa LaneNetPlus en lugar de LaneNet estándar
- `--use_attention`: Activa bloques de self-attention en el encoder
- `--use_multitask`: Habilita predicción de drivable area además de lanes
- `--use_rectification`: Aplica rectificación de homografía durante entrenamiento
- `--lambda_drivable`: Peso para la pérdida de drivable area (default: 0.5)
- `--drivable_dir`: Directorio con máscaras de drivable area
- `--save_visualizations`: Guarda visualizaciones durante entrenamiento

### Inferencia

- `--use_lanenet_plus`: Debe coincidir con el modelo entrenado
- `--use_attention`: Debe coincidir con el modelo entrenado
- `--use_multitask`: Debe coincidir con el modelo entrenado
- `--use_rectification`: Aplica rectificación durante inferencia

## Salidas del Modelo

LaneNetPlus retorna un diccionario con:

- `lane_logits`: Logits raw para máscara de carriles
- `lane_pred`: Predicción binaria de carriles
- `lane_prob`: Probabilidades de carriles
- `drivable_logits`: Logits raw para área transitable (si `use_multitask=True`)
- `drivable_pred`: Predicción binaria de área transitable (si `use_multitask=True`)
- `drivable_prob`: Probabilidades de área transitable (si `use_multitask=True`)
- `instance_seg_logits`: Segmentación de instancias (compatibilidad)
- `binary_seg_logits`: Logits de segmentación binaria (compatibilidad)
- `binary_seg_pred`: Predicción de segmentación binaria (compatibilidad)

## Métricas de Entrenamiento

El entrenamiento guarda las siguientes métricas en `training_log.csv`:

- `epoch`: Número de época
- `training_loss`: Pérdida total de entrenamiento
- `training_lane_loss`: Pérdida de carriles (entrenamiento)
- `training_instance_loss`: Pérdida de instancias (entrenamiento)
- `training_drivable_loss`: Pérdida de área transitable (entrenamiento, si multi-task)
- `val_loss`: Pérdida total de validación
- `val_lane_loss`: Pérdida de carriles (validación)
- `val_instance_loss`: Pérdida de instancias (validación)
- `val_drivable_loss`: Pérdida de área transitable (validación, si multi-task)

## Archivos Creados/Modificados

### Nuevos Archivos

- `model/lanenet/attention.py`: Módulo de self-attention
- `model/lanenet/multitask_lanenet.py`: Implementación multi-task
- `model/lanenet/LaneNetPlus.py`: Modelo principal integrado
- `model/lanenet/train_lanenet_plus.py`: Función de entrenamiento extendida
- `preprocess/homography_rectification.py`: Rectificación de homografía
- `preprocess/rectify_dataset.py`: Script de preprocesamiento de datasets
- `preprocess/__init__.py`: Inicialización del módulo

### Archivos Modificados

- `model/lanenet/loss.py`: Agregada `MultiTaskLoss`
- `dataloader/data_loaders.py`: Soporte para máscaras de drivable area
- `train.py`: Integración de LaneNetPlus
- `test.py`: Inferencia con todas las mejoras
- `model/utils/cli_helper.py`: Nuevos argumentos de línea de comandos
- `model/utils/cli_helper_test.py`: Nuevos argumentos para test

## Compatibilidad

- **Retrocompatibilidad**: El código original de LaneNet sigue funcionando sin cambios
- **Modelos existentes**: Los checkpoints de LaneNet estándar no son compatibles con LaneNetPlus
- **Flags**: Todos los flags son opcionales; el comportamiento por defecto es igual al original

## Ejemplos de Uso

### Ejemplo 1: Entrenar solo con multi-task
```bash
python train.py \
    --dataset data/training_data_example \
    --use_lanenet_plus \
    --use_multitask \
    --drivable_dir data/training_data_example/gt_image_drivable \
    --model_type ENet \
    --epochs 25
```

### Ejemplo 2: Entrenar con todas las mejoras
```bash
python train.py \
    --dataset data/training_data_example \
    --use_lanenet_plus \
    --use_attention \
    --use_multitask \
    --use_rectification \
    --drivable_dir data/training_data_example/gt_image_drivable \
    --model_type ENet \
    --epochs 50 \
    --save_visualizations
```

### Ejemplo 3: Inferencia con todas las características
```bash
python test.py \
    --img data/test_image.jpg \
    --model log/lanenet_plus/best_model.pth \
    --use_lanenet_plus \
    --use_attention \
    --use_multitask \
    --use_rectification \
    --model_type ENet
```

## Notas Importantes

1. **Coherencia de flags**: Los flags `--use_attention`, `--use_multitask` deben coincidir entre entrenamiento e inferencia
2. **Máscaras de drivable**: Si no existen, se generan placeholders vacíos. Agrega las máscaras reales cuando estén disponibles
3. **Rectificación**: La rectificación puede cambiar las dimensiones de las imágenes. Asegúrate de que el modelo esté entrenado con el mismo preprocesamiento
4. **Memoria**: El uso de atención aumenta el uso de memoria. Considera reducir el batch size si encuentras problemas

## Troubleshooting

### Error: "No lane/binary prediction found"
- Asegúrate de usar `--use_lanenet_plus` si el modelo fue entrenado con LaneNetPlus

### Error: "Drivable masks not found"
- Esto es normal si aún no has agregado las máscaras. El dataloader creará placeholders vacíos

### Error: "Homography rectification not available"
- Verifica que `preprocess/homography_rectification.py` esté en el path correcto

## Contribuciones

Este código extiende el proyecto original manteniendo compatibilidad hacia atrás. Todas las funciones originales siguen disponibles.

