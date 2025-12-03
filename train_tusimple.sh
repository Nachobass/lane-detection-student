#!/bin/bash

# Script para procesar y entrenar con dataset TUSimple
# Uso: ./train_tusimple.sh

set -e  # Salir si hay errores

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

# Comando de entrenamiento (SIN multitask porque TUSimple no tiene drivable area)
python train.py \
    --dataset "${TRAINING_DIR}" \
    --use_lanenet_plus \
    --use_attention \
    --use_rectification \
    --model_type ENet \
    --epochs 10 \
    --bs 4 \
    --lr 0.0001 \
    --save ./log/tusimple_lanenet_plus

echo ""
echo "========================================="
echo "Entrenamiento completado!"
echo "========================================="
echo "Modelo guardado en: ./log/tusimple_lanenet_plus/best_model.pth"
echo ""

