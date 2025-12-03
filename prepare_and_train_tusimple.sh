#!/bin/bash

# Script para preparar y entrenar con el dataset TUSimple
# Este script procesa el dataset JSON crudo y luego ejecuta el entrenamiento

TUSIMPLE_DIR="./data/archive/TUSimple/train_set"
TRAINING_DIR="${TUSIMPLE_DIR}/training"

echo "========================================="
echo "Preparando dataset TUSimple para entrenamiento"
echo "========================================="
echo "Directorio fuente: ${TUSIMPLE_DIR}"
echo "Directorio destino: ${TRAINING_DIR}"
echo ""

# Verificar si ya está procesado
if [ -d "${TRAINING_DIR}" ] && [ -f "${TRAINING_DIR}/train.txt" ] && [ -f "${TRAINING_DIR}/val.txt" ]; then
    echo "✓ El dataset ya está procesado en ${TRAINING_DIR}"
    echo "  - train.txt encontrado ($(wc -l < ${TRAINING_DIR}/train.txt) líneas)"
    echo "  - val.txt encontrado ($(wc -l < ${TRAINING_DIR}/val.txt) líneas)"
    echo ""
    read -p "¿Deseas re-procesar el dataset? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Usando dataset existente..."
        echo ""
    else
        echo "Eliminando dataset anterior y re-procesando..."
        rm -rf "${TRAINING_DIR}"
    fi
fi

# Procesar el dataset si no existe o si el usuario quiere re-procesar
if [ ! -d "${TRAINING_DIR}" ] || [ ! -f "${TRAINING_DIR}/train.txt" ]; then
    echo "Procesando dataset TUSimple..."
    echo "Esto puede tardar varios minutos..."
    echo ""
    
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
    else
        echo ""
        echo "✗ Error al procesar el dataset"
        exit 1
    fi
fi

# Verificar que los archivos existen
if [ ! -f "${TRAINING_DIR}/train.txt" ] || [ ! -f "${TRAINING_DIR}/val.txt" ]; then
    echo "✗ Error: No se encontraron train.txt o val.txt"
    exit 1
fi

echo "========================================="
echo "Comando de entrenamiento preparado"
echo "========================================="
echo ""
echo "Ejecuta el siguiente comando para entrenar:"
echo ""
echo "python train.py \\"
echo "    --dataset ${TRAINING_DIR} \\"
echo "    --use_lanenet_plus \\"
echo "    --use_attention \\"
echo "    --use_multitask \\"
echo "    --use_rectification \\"
echo "    --model_type ENet \\"
echo "    --epochs 50"
echo ""
echo "Nota: Si no tienes máscaras de drivable area, elimina --use_multitask y --drivable_dir"
echo ""

