import sounddevice as sd
from scipy.io.wavfile import write
import os
import re

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA GRABACIÓN
# ---------------------------------------------------------

# Frecuencia de muestreo.
# La práctica pide grabar a 16 kHz, por eso usamos 16000 Hz.
fs = 16000

# Duración de cada grabación en segundos.
# En este caso se grabará 1 segundo por archivo.
duracion = 1

# Palabra que se desea grabar.
# Se puede cambiar por otra palabra del conjunto,
# por ejemplo: "inicio", "alto", "pausa", "ruta", etc.
palabra = "alto"

# ---------------------------------------------------------
# CREACIÓN DE LA CARPETA PARA GUARDAR LAS GRABACIONES
# ---------------------------------------------------------

# Se crea una carpeta con el nombre de la palabra si no existe.
# Esto permite organizar las grabaciones por clases o categorías.
os.makedirs(palabra, exist_ok=True)

# ---------------------------------------------------------
# BÚSQUEDA DE GRABACIONES PREVIAS
# ---------------------------------------------------------

# Se listan los archivos que ya existen dentro de la carpeta.
archivos = os.listdir(palabra)

# Aquí se almacenarán los números de las grabaciones existentes.
numeros = []

# Se recorre cada archivo para detectar cuáles siguen el formato:
# palabra_numero.wav
# Ejemplo: sigue_1.wav, sigue_2.wav, sigue_3.wav
for archivo in archivos:
    match = re.match(rf"{palabra}_(\d+)\.wav", archivo)
    if match:
        numeros.append(int(match.group(1)))

# ---------------------------------------------------------
# GENERAR EL SIGUIENTE NOMBRE DE ARCHIVO
# ---------------------------------------------------------

# Si ya existen archivos, se toma el número más grande y se suma 1.
# Si no existe ninguno, se empieza desde 1.
siguiente_numero = max(numeros) + 1 if numeros else 1

# Se construye el nombre final del archivo.
# Ejemplo: sigue/sigue_4.wav
nombre_archivo = f"{palabra}/{palabra}_{siguiente_numero}.wav"

# ---------------------------------------------------------
# GRABACIÓN DE AUDIO
# ---------------------------------------------------------

print("Grabando...")

# sd.rec graba el audio desde el micrófono.
# int(duracion * fs) indica la cantidad total de muestras a capturar.
# samplerate=fs fija la frecuencia de muestreo en 16 kHz.
# channels=1 indica que se grabará en mono.
# dtype='float32' guarda las muestras en punto flotante.
audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float32')

# Espera a que termine la grabación antes de continuar.
sd.wait()

print("Grabación terminada.")

# ---------------------------------------------------------
# GUARDAR EL ARCHIVO WAV
# ---------------------------------------------------------

# Se guarda la grabación en formato WAV con la frecuencia de muestreo definida.
write(nombre_archivo, fs, audio)

# Se muestra la ruta del archivo generado.
print(f"Archivo guardado como: {nombre_archivo}")