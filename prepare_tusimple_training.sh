#!/bin/bash

# Script para preparar el dataset TUSimple para entrenamiento
# Este script procesa el dataset JSON crudo y genera las máscaras necesarias

TUSIMPLE_DIR="./data/archive/TUSimple/train_set"
TRAINING_DIR="${TUSIMPLE_DIR}/training"

echo "Preparando dataset TUSimple para entrenamiento..."
echo "Directorio fuente: ${TUSIMPLE_DIR}"
echo "Directorio destino: ${TRAINING_DIR}"

# Verificar si ya está procesado
if [ -d "${TRAINING_DIR}" ] && [ -f "${TRAINING_DIR}/train.txt" ] && [ -f "${TRAINING_DIR}/val.txt" ]; then
    echo "✓ El dataset ya está procesado en ${TRAINING_DIR}"
    echo "  - train.txt encontrado"
    echo "  - val.txt encontrado"
    echo ""
    echo "Para re-procesar, elimina la carpeta ${TRAINING_DIR}"
    exit 0
fi

# Procesar el dataset
echo "Procesando dataset TUSimple..."
python tusimple_transform.py \
    --src_dir "${TUSIMPLE_DIR}" \
    --val True \
    --test False

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Dataset procesado exitosamente!"
    echo "  - Archivos train.txt y val.txt generados en ${TRAINING_DIR}"
    echo "  - Máscaras generadas en ${TRAINING_DIR}/gt_*"
    echo ""
    echo "Ahora puedes ejecutar el entrenamiento con:"
    echo "  python train.py --dataset ${TRAINING_DIR} [otros parámetros]"
else
    echo ""
    echo "✗ Error al procesar el dataset"
    exit 1
fi

