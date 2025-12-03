# Guía para Entrenar con Dataset TUSimple

## Estructura del Dataset

Tu dataset TUSimple está en:
```
data/archive/TUSimple/train_set/
├── clips/
│   ├── 0313-1/
│   ├── 0313-2/
│   ├── 0531/
│   └── 0601/
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

## Paso 1: Procesar el Dataset

Primero necesitas procesar los archivos JSON para generar las máscaras binarias e instance, y los archivos `train.txt` y `val.txt`:

```bash
python tusimple_transform.py \
    --src_dir ./data/archive/TUSimple/train_set \
    --val True \
    --test False
```

Esto creará:
- `data/archive/TUSimple/train_set/training/gt_image/` - Imágenes procesadas
- `data/archive/TUSimple/train_set/training/gt_binary_image/` - Máscaras binarias
- `data/archive/TUSimple/train_set/training/gt_instance_image/` - Máscaras instance
- `data/archive/TUSimple/train_set/training/train.txt` - Lista de entrenamiento
- `data/archive/TUSimple/train_set/training/val.txt` - Lista de validación

**Nota:** Este proceso puede tardar varios minutos dependiendo del tamaño del dataset.

## Paso 2: Entrenar el Modelo

### Opción A: Sin Multi-task (Recomendado para empezar)

Como el dataset TUSimple no incluye máscaras de drivable area, usa este comando:

```bash
python train.py \
    --dataset ./data/archive/TUSimple/train_set/training \
    --use_lanenet_plus \
    --use_attention \
    --use_rectification \
    --model_type ENet \
    --epochs 50 \
    --bs 4 \
    --lr 0.0001 \
    --save ./log/tusimple_lanenet_plus
```

### Opción B: Con Multi-task (Solo si tienes máscaras drivable)

Si más adelante quieres agregar máscaras de drivable area:

```bash
python train.py \
    --dataset ./data/archive/TUSimple/train_set/training \
    --use_lanenet_plus \
    --use_attention \
    --use_multitask \
    --use_rectification \
    --drivable_dir ./data/archive/TUSimple/train_set/training/gt_image_drivable \
    --model_type ENet \
    --epochs 50 \
    --bs 4 \
    --lr 0.0001 \
    --save ./log/tusimple_lanenet_plus_multitask
```

## Parámetros del Comando

- `--dataset`: Ruta al directorio que contiene `train.txt` y `val.txt`
- `--use_lanenet_plus`: Usa LaneNetPlus en lugar de LaneNet estándar
- `--use_attention`: Activa bloques de self-attention en el encoder
- `--use_multitask`: Habilita predicción de drivable area (requiere máscaras)
- `--use_rectification`: Aplica rectificación de homografía durante entrenamiento
- `--drivable_dir`: Directorio con máscaras de drivable area (solo si usas `--use_multitask`)
- `--model_type`: Tipo de backbone (`ENet`, `UNet`, o `DeepLabv3+`)
- `--epochs`: Número de épocas de entrenamiento
- `--bs`: Batch size (ajusta según tu GPU)
- `--lr`: Learning rate
- `--save`: Directorio donde guardar el modelo entrenado

## Usar el Script Automático

También puedes usar el script `prepare_and_train_tusimple.sh` que procesa el dataset y te muestra el comando correcto:

```bash
chmod +x prepare_and_train_tusimple.sh
./prepare_and_train_tusimple.sh
```

## Verificar el Procesamiento

Para verificar que el dataset está correctamente procesado:

```bash
# Ver cantidad de imágenes de entrenamiento
wc -l ./data/archive/TUSimple/train_set/training/train.txt

# Ver cantidad de imágenes de validación
wc -l ./data/archive/TUSimple/train_set/training/val.txt

# Ver una línea de ejemplo
head -1 ./data/archive/TUSimple/train_set/training/train.txt
```

El formato esperado es:
```
/path/to/image/0000.png /path/to/gt_binary_image/0000.png /path/to/gt_instance_image/0000.png
```

