#!/bin/bash

# Script para evaluar el modelo entrenado y calcular F1 score
# Este script evalúa el modelo en el conjunto de validación

MODEL_PATH="./log/tusimple_lanenet_plus/best_model.pth"
DATASET_DIR="./data/archive_train/TUSimple/train_set/training"

echo "========================================="
echo "Evaluando modelo entrenado"
echo "========================================="
echo "Modelo: ${MODEL_PATH}"
echo "Dataset: ${DATASET_DIR}"
echo ""

# Verificar que el modelo existe
if [ ! -f "${MODEL_PATH}" ]; then
    echo "✗ Error: Modelo no encontrado en ${MODEL_PATH}"
    exit 1
fi

# Verificar que el dataset existe
if [ ! -f "${DATASET_DIR}/val.txt" ]; then
    echo "✗ Error: Archivo val.txt no encontrado en ${DATASET_DIR}"
    exit 1
fi

echo "Evaluando en conjunto de validación..."
echo ""

# Evaluar en validación
python eval_model.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_DIR}" \
    --split val \
    --use_lanenet_plus \
    --use_attention \
    --model_type ENet \
    --save_csv ./log/tusimple_lanenet_plus/eval_results_val.csv \
    --verbose

echo ""
echo "========================================="
echo "Evaluación completada!"
echo "========================================="

