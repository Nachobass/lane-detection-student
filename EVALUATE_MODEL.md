# Evaluar Modelo Entrenado - Calcular F1 Score e IoU

## Scripts Creados

1. **`eval_model.py`** - Script principal de evaluación
2. **`eval_best_model.sh`** - Script de shell para ejecutar fácilmente

## Uso Rápido

### Opción 1: Script Automático (Recomendado)

```bash
chmod +x eval_best_model.sh
./eval_best_model.sh
```

Este script:
- Evalúa el modelo en el conjunto de validación
- Calcula F1 Score (Dice) e IoU promedio
- Guarda resultados detallados en CSV

### Opción 2: Comando Manual

```bash
python eval_model.py \
    --model ./log/tusimple_lanenet_plus/best_model.pth \
    --dataset ./data/archive_train/TUSimple/train_set/training \
    --split val \
    --use_lanenet_plus \
    --use_attention \
    --model_type ENet \
    --save_csv ./log/tusimple_lanenet_plus/eval_results_val.csv \
    --verbose
```

## Parámetros del Script

### Requeridos
- `--model`: Ruta al modelo entrenado (archivo .pth)
- `--dataset`: Directorio que contiene train.txt/val.txt/test.txt

### Opcionales
- `--split`: Qué conjunto evaluar (`train`, `val`, o `test`) - Default: `val`
- `--model_type`: Arquitectura del backbone (`ENet`, `UNet`, `DeepLabv3+`) - Default: `ENet`
- `--use_lanenet_plus`: Activar si el modelo es LaneNetPlus
- `--use_attention`: Activar si se usó attention en entrenamiento
- `--use_multitask`: Activar si se usó multi-task en entrenamiento
- `--height`: Altura de resize - Default: 256
- `--width`: Ancho de resize - Default: 512
- `--bs`: Batch size - Default: 1
- `--save_csv`: Guardar resultados detallados en CSV (ruta opcional)
- `--verbose`: Mostrar resultados por cada imagen

## Ejemplos

### Evaluar en Validación (sin verbose)
```bash
python eval_model.py \
    --model ./log/tusimple_lanenet_plus/best_model.pth \
    --dataset ./data/archive_train/TUSimple/train_set/training \
    --split val \
    --use_lanenet_plus \
    --use_attention \
    --model_type ENet
```

### Evaluar en Test Set
```bash
python eval_model.py \
    --model ./log/tusimple_lanenet_plus/best_model.pth \
    --dataset ./data/archive_train/TUSimple/train_set/training \
    --split test \
    --use_lanenet_plus \
    --use_attention \
    --model_type ENet \
    --save_csv ./log/tusimple_lanenet_plus/eval_results_test.csv
```

### Evaluar Modelo LaneNet Estándar
```bash
python eval_model.py \
    --model ./log/best_model.pth \
    --dataset ./data/training_data_example \
    --split val \
    --model_type ENet
```

## Resultados

El script muestra:
- **Average IoU**: Intersection over Union promedio
- **Average F1 (Dice)**: F1 Score (coeficiente de Dice) promedio
- **Total images evaluated**: Cantidad de imágenes procesadas
- **Evaluation time**: Tiempo total y por imagen

Si se usa `--save_csv`, también guarda:
- Resultados por imagen (IoU y F1 individuales)
- Estadísticas (min, max, std)

## Notas

- El script carga el modelo con `map_location=DEVICE` para funcionar en CPU o GPU
- Las métricas se calculan usando la misma función `Eval_Score` que se usa durante el entrenamiento
- F1 Score = Dice Coefficient = `2 * intersection / (sum_pred + sum_true)`
- IoU = `intersection / union`

