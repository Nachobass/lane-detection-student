#!/bin/bash

# Script para procesar y entrenar con dataset TUSimple
# Uso: ./train_tusimple.sh

# No usar set -e porque queremos continuar aunque la evaluación falle

# TUSIMPLE_DIR="./data/archive/TUSimple/train_set"
TUSIMPLE_DIR="./data/archive_train/TUSimple/train_set"
TRAINING_DIR="${TUSIMPLE_DIR}/training"

echo "========================================="
echo "Entrenamiento con Dataset TUSimple"
echo "========================================="
echo ""

# Paso 1: Verificar/Procesar dataset
if [ ! -d "${TRAINING_DIR}" ] || [ ! -f "${TRAINING_DIR}/train.txt" ]; then
    echo "Paso 1: Procesando dataset TUSimple..."
    echo "Esto puede tardar varios minutos..."
    echo ""
    
    python tusimple_transform.py \
        --src_dir "${TUSIMPLE_DIR}" \
        --val True \
        --test False
    
    if [ $? -ne 0 ]; then
        echo "✗ Error al procesar el dataset"
        exit 1
    fi
    
    echo ""
    echo "✓ Dataset procesado exitosamente!"
    echo ""
else
    echo "Paso 1: Dataset ya procesado ✓"
    echo "  - train.txt: $(wc -l < ${TRAINING_DIR}/train.txt) líneas"
    echo "  - val.txt: $(wc -l < ${TRAINING_DIR}/val.txt) líneas"
    echo ""
fi

# Verificar que los archivos existen
if [ ! -f "${TRAINING_DIR}/train.txt" ] || [ ! -f "${TRAINING_DIR}/val.txt" ]; then
    echo "✗ Error: No se encontraron train.txt o val.txt"
    exit 1
fi

# Paso 2: Entrenar
echo "========================================="
echo "Paso 2: Iniciando entrenamiento"
echo "========================================="
echo ""

# Verificar si existen pesos pre-entrenados
PRETRAINED_WEIGHTS="./log/baseline_weights/best_model.pth"
if [ -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "✓ Pesos pre-entrenados encontrados: ${PRETRAINED_WEIGHTS}"
    echo "  El entrenamiento comenzará desde estos pesos"
    PRETRAINED_ARG="--pretrained ${PRETRAINED_WEIGHTS}"
else
    echo "⚠ No se encontraron pesos pre-entrenados en ${PRETRAINED_WEIGHTS}"
    echo "  El entrenamiento comenzará desde cero"
    PRETRAINED_ARG=""
fi
echo ""

# Comando de entrenamiento (SIN multitask porque TUSimple no tiene drivable area)
python train.py \
    --dataset "${TRAINING_DIR}" \
    --use_lanenet_plus \
    --use_attention \
    --use_rectification \
    --model_type ENet \
    --epochs 25 \
    --bs 4 \
    --lr 0.0001 \
    --save ./log/tusimple_lanenet_plus \
    ${PRETRAINED_ARG}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "✗ Error durante el entrenamiento"
    exit 1
fi

echo ""
echo "========================================="
echo "Entrenamiento completado!"
echo "========================================="
echo "Modelo guardado en: ./log/tusimple_lanenet_plus/best_model.pth"
echo ""

# Paso 3: Evaluar y calcular F1
echo "========================================="
echo "Paso 3: Evaluando modelo y calculando F1"
echo "========================================="
echo ""

python eval_model.py \
    --model ./log/tusimple_lanenet_plus/best_model.pth \
    --dataset "${TRAINING_DIR}" \
    --split val \
    --use_lanenet_plus \
    --use_attention \
    --model_type ENet \
    --save_csv ./log/tusimple_lanenet_plus/eval_results_val.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Evaluación completada!"
    echo "========================================="
    echo "Resultados guardados en: ./log/tusimple_lanenet_plus/eval_results_val.csv"
else
    echo ""
    echo "⚠ Error durante la evaluación (continúa el script)"
fi

echo ""
















# #!/bin/bash

# # Script para procesar y entrenar con dataset TUSimple
# # Uso: ./train_tusimple.sh

# # No usar set -e porque queremos continuar aunque la evaluación falle

# # TUSIMPLE_DIR="./data/archive/TUSimple/train_set"
# TUSIMPLE_DIR="./data/archive_train/TUSimple/train_set"
# TRAINING_DIR="${TUSIMPLE_DIR}/training"

# echo "========================================="
# echo "Entrenamiento con Dataset TUSimple"
# echo "========================================="
# echo ""

# # Paso 1: Verificar/Procesar dataset
# if [ ! -d "${TRAINING_DIR}" ] || [ ! -f "${TRAINING_DIR}/train.txt" ]; then
#     echo "Paso 1: Procesando dataset TUSimple..."
#     echo "Esto puede tardar varios minutos..."
#     echo ""
    
#     python tusimple_transform.py \
#         --src_dir "${TUSIMPLE_DIR}" \
#         --val True \
#         --test False
    
#     if [ $? -ne 0 ]; then
#         echo "✗ Error al procesar el dataset"
#         exit 1
#     fi
    
#     echo ""
#     echo "✓ Dataset procesado exitosamente!"
#     echo ""
# else
#     echo "Paso 1: Dataset ya procesado ✓"
#     echo "  - train.txt: $(wc -l < ${TRAINING_DIR}/train.txt) líneas"
#     echo "  - val.txt: $(wc -l < ${TRAINING_DIR}/val.txt) líneas"
#     echo ""
# fi

# # Verificar que los archivos existen
# if [ ! -f "${TRAINING_DIR}/train.txt" ] || [ ! -f "${TRAINING_DIR}/val.txt" ]; then
#     echo "✗ Error: No se encontraron train.txt o val.txt"
#     exit 1
# fi

# # Paso 2: Entrenar
# echo "========================================="
# echo "Paso 2: Iniciando entrenamiento"
# echo "========================================="
# echo ""

# # Verificar si existen pesos pre-entrenados
# PRETRAINED_WEIGHTS="./log/baseline_weights/best_model.pth"
# if [ -f "${PRETRAINED_WEIGHTS}" ]; then
#     echo "✓ Pesos pre-entrenados encontrados: ${PRETRAINED_WEIGHTS}"
#     echo "  El entrenamiento comenzará desde estos pesos"
#     PRETRAINED_ARG="--pretrained ${PRETRAINED_WEIGHTS}"
# else
#     echo "⚠ No se encontraron pesos pre-entrenados en ${PRETRAINED_WEIGHTS}"
#     echo "  El entrenamiento comenzará desde cero"
#     PRETRAINED_ARG=""
# fi
# echo ""

# # Comando de entrenamiento con LaneNet estándar (sin LaneNetPlus)
# python train.py \
#     --dataset "${TRAINING_DIR}" \
#     --model_type ENet \
#     --epochs 25 \
#     --bs 4 \
#     --lr 0.0001 \
#     --save ./log/tusimple_lanenet \
#     ${PRETRAINED_ARG}

# TRAIN_EXIT_CODE=$?
# if [ $TRAIN_EXIT_CODE -ne 0 ]; then
#     echo "✗ Error durante el entrenamiento"
#     exit 1
# fi

# echo ""
# echo "========================================="
# echo "Entrenamiento completado!"
# echo "========================================="
# echo "Modelo guardado en: ./log/tusimple_lanenet/best_model.pth"
# echo ""

# # Paso 3: Evaluar y calcular F1
# echo "========================================="
# echo "Paso 3: Evaluando modelo y calculando F1"
# echo "========================================="
# echo ""

# python eval_model.py \
#     --model ./log/tusimple_lanenet/best_model.pth \
#     --dataset "${TRAINING_DIR}" \
#     --split val \
#     --model_type ENet \
#     --save_csv ./log/tusimple_lanenet/eval_results_val.csv

# if [ $? -eq 0 ]; then
#     echo ""
#     echo "========================================="
#     echo "Evaluación completada!"
#     echo "========================================="
#     echo "Resultados guardados en: ./log/tusimple_lanenet/eval_results_val.csv"
# else
#     echo ""
#     echo "⚠ Error durante la evaluación (continúa el script)"
# fi

# echo ""

