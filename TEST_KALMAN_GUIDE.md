# Guía para Probar el Filtro de Kalman

## ¿Por qué necesitas múltiples imágenes?

El filtro de Kalman necesita **historial temporal** para funcionar correctamente:
- **Primer frame**: Solo inicializa, usa 100% ENet
- **Segundos frames**: Comienza a fusionar predicciones con detecciones
- **Frames siguientes**: El Kalman aprende el movimiento y suaviza las detecciones

## Cómo usar el script de secuencia

### Opción 1: Procesar una carpeta con imágenes

Si tienes todas las imágenes en una carpeta (por ejemplo, `data/test_kalman/`):

```bash
python test_sequence.py \
    --img_dir ./data/test_kalman \
    --use_kalman \
    --freeze_backbone \
    --output ./test_output_sequence
```

### Opción 2: Usar un patrón de archivos

```bash
python test_sequence.py \
    --img_dir "./data/test_kalman/*.jpg" \
    --use_kalman \
    --freeze_backbone \
    --kalman_process_noise 0.03 \
    --kalman_measurement_noise 0.3 \
    --max_lanes 3 \
    --output ./test_output_sequence
```

### Parámetros disponibles

- `--img_dir`: Directorio con imágenes o patrón (ej: `./data/test_kalman` o `./data/test_kalman/*.jpg`)
- `--use_kalman`: Activa el filtro de Kalman (OBLIGATORIO para testing)
- `--freeze_backbone`: Congela el encoder
- `--kalman_process_noise`: Ruido del proceso (default: 0.03)
- `--kalman_measurement_noise`: Ruido de medición (default: 0.3)
- `--max_lanes`: Máximo número de carriles a trackear (default: 4)
- `--output`: Directorio donde guardar resultados (default: `./test_output_sequence`)

## Estructura de salida

El script crea una estructura organizada:

```
test_output_sequence/
├── frame_0001/
│   ├── input_overlay.jpg      # Imagen original con líneas superpuestas
│   ├── binary_mask.jpg        # Máscara binaria de detección
│   └── instance_mask.jpg      # Máscara de instancias (si está disponible)
├── frame_0002/
│   ├── input_overlay.jpg
│   ├── binary_mask.jpg
│   └── instance_mask.jpg
├── frame_0003/
│   └── ...
└── ...
```

Cada frame muestra en la esquina superior izquierda:
- Número de frame
- Peso del Kalman usado
- Número de carriles trackeados

## Ejemplo completo

```bash
# Procesar todas las imágenes en test_kalman con Kalman
python test_sequence.py \
    --img_dir ./data/test_kalman \
    --use_kalman \
    --freeze_backbone \
    --kalman_process_noise 0.03 \
    --kalman_measurement_noise 0.1 \
    --max_lanes 3 \
    --output ./kalman_results
```

## Qué observar en los resultados

### Frame 1 (Primera imagen)
- **Sin Kalman visible**: Solo se inicializa
- Uso: 100% ENet, 0% Kalman
- Deberías ver las detecciones de ENet sin cambios

### Frame 2-5 (Primeras imágenes)
- **Kalman comienza a funcionar**: Empieza a suavizar
- Uso: ~90% ENet, ~10% Kalman
- Las líneas deberían empezar a suavizarse ligeramente

### Frame 6+ (Imágenes siguientes)
- **Kalman activo**: Predicciones y suavizado
- Uso: Depende de la confianza en ENet
- Las líneas deberían ser más suaves y estables

### Indicadores de que funciona bien:
✅ Las líneas son más suaves entre frames  
✅ Hay menos "saltos" o jitter  
✅ Si ENet falla temporalmente, Kalman mantiene el tracking  
✅ Las líneas muestran continuidad temporal

### Problemas a detectar:
❌ Las líneas se vuelven muy "suaves" y pierden precisión  
❌ Las líneas no se actualizan cuando ENet detecta cambios  
❌ Formas extrañas o carriles que se cruzan

## Comparar con y sin Kalman

### Con Kalman:
```bash
python test_sequence.py --img_dir ./data/test_kalman --use_kalman --output ./results_with_kalman
```

### Sin Kalman (solo ENet):
```bash
python test_sequence.py --img_dir ./data/test_kalman --output ./results_no_kalman
```

Luego compara las carpetas `results_with_kalman` y `results_no_kalman` para ver las diferencias.

## Ajustar parámetros del Kalman

### Si las líneas son muy "rígidas" (no siguen bien los cambios):
- **Aumenta** `--kalman_process_noise` (ej: 0.05 o 0.1)
- Esto hace que el Kalman confíe menos en sus predicciones

### Si las líneas tienen mucho ruido/jitter:
- **Disminuye** `--kalman_process_noise` (ej: 0.01)
- **Aumenta** `--kalman_measurement_noise` (ej: 0.5)
- Esto hace que el Kalman confíe más en sus predicciones y suavice más

### Si hay demasiadas detecciones falsas:
- **Aumenta** `--kalman_measurement_noise` (ej: 0.5)
- Esto hace que el Kalman confíe menos en ENet

## Notas importantes

1. **Orden de las imágenes**: El script procesa las imágenes en orden alfabético. Asegúrate de que los nombres de archivo reflejen el orden temporal (ej: `001.jpg`, `002.jpg`, etc.)

2. **Secuencia de video real**: Si tienes un video, primero extráelo a frames:
   ```bash
   # Usando ffmpeg
   ffmpeg -i video.mp4 -qscale:v 2 data/test_kalman/frame_%04d.jpg
   ```

3. **Primer frame**: El primer frame siempre usa solo ENet (esto es correcto, no hay historial aún)

4. **Velocidad**: Procesar secuencias puede ser más lento, especialmente si hay muchas imágenes

## Troubleshooting

### Error: "No images found"
- Verifica que la ruta del directorio sea correcta
- Verifica que haya archivos `.jpg`, `.png`, etc. en el directorio

### Las líneas no se suavizan
- Verifica que `--use_kalman` esté activado
- Verifica que estés procesando múltiples frames (no solo uno)
- Revisa los pesos en la esquina de cada imagen

### Mucho ruido o predicciones erróneas
- Ajusta `--kalman_process_noise` y `--kalman_measurement_noise`
- Asegúrate de que las imágenes estén en orden temporal correcto

