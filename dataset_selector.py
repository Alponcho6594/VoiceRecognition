import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# =========================================================
# CARGA Y PREPROCESAMIENTO
# =========================================================

def cargar_audio(ruta_audio):
    """
    Carga un archivo WAV y realiza un preprocesamiento básico.

    Qué hace:
    1. Lee el audio desde disco.
    2. Si el archivo tiene dos canales, lo convierte a mono.
    3. Convierte las muestras a tipo float32.
    4. Normaliza la amplitud.
    5. Elimina el offset DC.
    """
    fs, audio = wavfile.read(ruta_audio)

    # Si el audio es estéreo, se promedian los dos canales para obtener mono.
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
        print(f"[INFO] {os.path.basename(ruta_audio)} convertido a mono.")

    # Conversión a flotante para facilitar operaciones matemáticas.
    audio = audio.astype(np.float32)

    # Normalización por el máximo valor absoluto.
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Eliminación del valor medio de la señal.
    audio = audio - np.mean(audio)

    return fs, audio


def aplicar_preenfasis(audio, alpha=0.95):
    """
    Aplica el filtro de preénfasis.

    Fórmula:
        y[n] = x[n] - alpha * x[n-1]
    """
    audio_pre = np.empty_like(audio)
    audio_pre[0] = audio[0]
    audio_pre[1:] = audio[1:] - alpha * audio[:-1]
    return audio_pre


def dividir_en_tramas(audio, frame_length=320, hop_length=128):
    """
    Divide la señal en tramas traslapadas.
    """
    num_samples = len(audio)

    # Si el audio es más corto que una trama, se rellena con ceros.
    if num_samples < frame_length:
        padded = np.zeros(frame_length, dtype=audio.dtype)
        padded[:num_samples] = audio
        return padded[np.newaxis, :]

    # Número de tramas completas disponibles.
    num_frames = 1 + (num_samples - frame_length) // hop_length
    frames = np.zeros((num_frames, frame_length), dtype=audio.dtype)

    # Se van extrayendo bloques con el salto indicado.
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i, :] = audio[start:end]

    return frames


def crear_ventana_hamming(N=320):
    """
    Genera una ventana de Hamming de tamaño N.

    Relación con las PPTs:
    La ventana de Hamming se usa para suavizar los extremos de cada trama
    y evitar discontinuidades bruscas.
    """
    n = np.arange(N)
    w = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))
    return w.astype(np.float32)


def aplicar_ventana_hamming(frames):
    """
    Aplica una ventana de Hamming a cada trama.
    """
    N = frames.shape[1]
    ventana = crear_ventana_hamming(N)
    frames_windowed = frames * ventana
    return frames_windowed, ventana


def calcular_energia_por_trama(frames):
    """
    Calcula la energía promedio de cada trama.

    Esta medida ayuda a detectar regiones donde sí hay voz.
    """
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def calcular_zcr_por_trama(frames):
    """
    Calcula la tasa de cruces por cero (ZCR) para cada trama.

    Esta característica se usa como apoyo para segmentar la palabra.
    """
    zcr = np.zeros(frames.shape[0], dtype=np.float32)

    for i, frame in enumerate(frames):
        signos = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signos))) / 2
        zcr[i] = crossings / len(frame)

    return zcr


def detectar_inicio_fin(frames, hop_length, frame_length,
                        energy_factor=0.03, zcr_factor=0.08):
    """
    Detecta el inicio y el final de la región con voz.

    Estrategia:
    - se calcula energía por trama,
    - se calcula ZCR por trama,
    - se definen umbrales relativos,
    - se marca como voz la trama que supere ambos umbrales.

    Relación con la práctica:
    Esto sigue la idea del recorte de silencios usando potencia
    y cruces por cero.
    """
    energia = calcular_energia_por_trama(frames)
    zcr = calcular_zcr_por_trama(frames)

    energy_threshold = energy_factor * np.max(energia)
    zcr_threshold = zcr_factor * np.max(zcr)

    voice_flags = (zcr > zcr_threshold) & (energia > energy_threshold)
    voice_idx = np.where(voice_flags)[0]

    # Si no se detecta voz, se devuelve None.
    if len(voice_idx) == 0:
        return None, None, energia, zcr, voice_flags, energy_threshold, zcr_threshold

    # Primera y última trama con voz.
    inicio_frame = voice_idx[0]
    fin_frame = voice_idx[-1]

    # Conversión de índices de trama a índices de muestra.
    start_sample = inicio_frame * hop_length
    end_sample = fin_frame * hop_length + frame_length

    return start_sample, end_sample, energia, zcr, voice_flags, energy_threshold, zcr_threshold


def recortar_audio(audio, start_sample, end_sample):
    """
    Recorta el audio usando el inicio y final detectados.
    """
    if start_sample is None or end_sample is None:
        return None

    end_sample = min(end_sample, len(audio))
    return audio[start_sample:end_sample]


# =========================================================
# LPC Y LSF
# =========================================================

def autocorrelacion_corta(frame, order=12):
    """
    Calcula la autocorrelación corta de una trama.

    Relación con las PPTs:
    La autocorrelación es la base para el cálculo de LPC.
    """
    r = np.zeros(order + 1, dtype=np.float64)

    for k in range(order + 1):
        r[k] = np.sum(frame[:len(frame) - k] * frame[k:])

    return r


def levinson_durbin(r, order=12):
    """
    Implementa el algoritmo de Levinson-Durbin para obtener
    coeficientes LPC a partir de la autocorrelación.

    Relación con las PPTs:
    Esta es precisamente la técnica mostrada para resolver
    el sistema de ecuaciones del predictor lineal.
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    e = r[0]

    if e <= 1e-12:
        return a, e

    for i in range(1, order + 1):
        suma = 0.0
        for j in range(1, i):
            suma += a[j] * r[i - j]

        k = (r[i] - suma) / e

        a_new = a.copy()
        a_new[i] = k

        for j in range(1, i):
            a_new[j] = a[j] - k * a[i - j]

        a = a_new
        e = (1.0 - k * k) * e

        if e <= 1e-12:
            e = 1e-12
            break

    return a, e


def extraer_lpc_por_trama(frames_hamming, order=12):
    """
    Extrae LPC por cada trama ya enventanada.

    Para cada trama guarda:
    - autocorrelación,
    - coeficientes LPC,
    - error de predicción.
    """
    num_frames = frames_hamming.shape[0]

    lpc_vectors = np.zeros((num_frames, order + 1), dtype=np.float64)
    r_vectors = np.zeros((num_frames, order + 1), dtype=np.float64)
    errors = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        frame = frames_hamming[i]
        r = autocorrelacion_corta(frame, order=order)
        a, e = levinson_durbin(r, order=order)

        r_vectors[i, :] = r
        lpc_vectors[i, :] = a
        errors[i] = e

    return lpc_vectors, r_vectors, errors


def _unique_angles_sorted(angles, tol=1e-4):
    """
    Ordena ángulos y elimina valores casi duplicados.

    Esto ayuda a estabilizar la conversión LPC -> LSF.
    """
    if len(angles) == 0:
        return angles

    angles = np.sort(angles)
    unicos = [angles[0]]

    for ang in angles[1:]:
        if np.abs(ang - unicos[-1]) > tol:
            unicos.append(ang)

    return np.array(unicos)


def lpc_a_lsf(a):
    """
    Convierte un vector LPC a LSF.

    Relación con la práctica:
    El agrupamiento del sistema se hace sobre LSF, ya que estos parámetros
    son más adecuados y estables para clustering que los LPC directos.
    """
    a = np.asarray(a, dtype=np.float64)

    if a[0] == 0:
        raise ValueError("El primer coeficiente LPC no puede ser cero.")

    a = a / a[0]

    ar = np.concatenate(([1.0], -a[1:]))
    p = len(ar) - 1

    ar_padded = np.concatenate((ar, [0.0]))
    ar_rev = ar_padded[::-1]

    P = ar_padded + ar_rev
    Q = ar_padded - ar_rev

    P_red, _ = np.polydiv(P, np.array([1.0, 1.0]))
    Q_red, _ = np.polydiv(Q, np.array([1.0, -1.0]))

    P_red = np.real_if_close(P_red, tol=1000).astype(np.float64)
    Q_red = np.real_if_close(Q_red, tol=1000).astype(np.float64)

    roots_P = np.roots(P_red)
    roots_Q = np.roots(Q_red)

    ang_P = np.abs(np.angle(roots_P))
    ang_Q = np.abs(np.angle(roots_Q))

    ang = np.concatenate((ang_P, ang_Q))

    eps = 1e-6
    ang = ang[(ang > eps) & (ang < np.pi - eps)]

    ang = _unique_angles_sorted(ang, tol=1e-4)

    if len(ang) < p:
        raise ValueError(
            f"No se pudieron obtener suficientes LSF. Esperadas: {p}, obtenidas: {len(ang)}"
        )

    if len(ang) > p:
        ang = ang[:p]

    return ang


# =========================================================
# PIPELINE COMPLETO DE UN AUDIO
# =========================================================

def procesar_audio_a_caracteristicas(
    ruta_audio,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    orden_lpc=12,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65,
    graficar=False
):
    """
    Procesa un archivo de audio completo y devuelve tanto las señales
    intermedias como las características extraídas.

    Este pipeline sigue la misma lógica del sistema principal:
    1. carga del audio,
    2. preénfasis,
    3. segmentación en tramas,
    4. ventana de Hamming,
    5. detección de inicio y fin,
    6. recorte de la palabra,
    7. extracción de LPC,
    8. conversión a LSF.

    La diferencia es que aquí, además de extraer características,
    se guardan muchos resultados intermedios porque el objetivo es
    inspeccionar y evaluar la calidad del dataset.
    """
    fs, audio = cargar_audio(ruta_audio)
    nombre_archivo = os.path.basename(ruta_audio)

    # Aplicación de preénfasis.
    audio_pre = aplicar_preenfasis(audio, alpha=alpha)

    # Se generan tramas de la señal completa.
    frames = dividir_en_tramas(audio_pre, frame_length=frame_length, hop_length=hop_length)

    # Se aplica ventana de Hamming.
    frames_hamming, _ = aplicar_ventana_hamming(frames)

    # Se detecta la región con voz.
    resultado = detectar_inicio_fin(
        frames_hamming,
        hop_length=hop_length,
        frame_length=frame_length,
        energy_factor=energy_factor,
        zcr_factor=zcr_factor
    )

    start_sample, end_sample, energia, zcr, voice_flags, energy_threshold, zcr_threshold = resultado

    if start_sample is None or end_sample is None:
        raise ValueError(f"No se detectó voz en el archivo: {ruta_audio}")

    # Se añade un margen para evitar recortar demasiado agresivamente.
    margen = int((margen_ms / 1000.0) * fs)
    start_sample = max(0, start_sample - margen)
    end_sample = min(len(audio_pre), end_sample + margen)

    # Se recorta la región útil.
    audio_recortado = recortar_audio(audio_pre, start_sample, end_sample)

    # Se vuelve a segmentar solo la parte recortada.
    frames_rec = dividir_en_tramas(audio_recortado, frame_length=frame_length, hop_length=hop_length)
    frames_rec_hamming, _ = aplicar_ventana_hamming(frames_rec)

    # Se extraen LPC, autocorrelaciones y errores.
    lpc, r, errors = extraer_lpc_por_trama(frames_rec_hamming, order=orden_lpc)

    # Se convierten los vectores LPC a LSF.
    lsf = []
    indices_validos = []

    for i, a in enumerate(lpc):
        try:
            lsf_i = lpc_a_lsf(a)
            lsf.append(lsf_i)
            indices_validos.append(i)
        except Exception:
            continue

    if len(lsf) == 0:
        raise ValueError(f"No se pudieron obtener vectores LSF en: {ruta_audio}")

    lsf = np.array(lsf, dtype=np.float64)
    indices_validos = np.array(indices_validos, dtype=int)

    lpc = lpc[indices_validos]
    r = r[indices_validos]
    errors = errors[indices_validos]

    # Opción para visualizar audio original y audio recortado.
    if graficar:
        t = np.arange(len(audio)) / fs
        t_rec = np.arange(len(audio_recortado)) / fs

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, audio)
        plt.axvline(start_sample / fs, color='g', linestyle='--', label='Inicio')
        plt.axvline(end_sample / fs, color='r', linestyle='--', label='Fin')
        plt.title(f"Audio original - {nombre_archivo}")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_rec, audio_recortado)
        plt.title(f"Audio recortado - {nombre_archivo}")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return {
        "ruta": ruta_audio,
        "nombre": nombre_archivo,
        "fs": fs,
        "audio_original": audio,
        "audio_pre": audio_pre,
        "audio_recortado": audio_recortado,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "energia": energia,
        "zcr": zcr,
        "voice_flags": voice_flags,
        "energy_threshold": energy_threshold,
        "zcr_threshold": zcr_threshold,
        "frames": frames,
        "frames_hamming": frames_hamming,
        "frames_rec": frames_rec,
        "frames_rec_hamming": frames_rec_hamming,
        "lpc": lpc,
        "r": r,
        "errors": errors,
        "lsf": lsf,
        "num_frames_total": len(frames_rec),
        "num_frames_validas": len(lsf)
    }


# =========================================================
# MÉTRICAS DE CALIDAD
# =========================================================

def calcular_rms(audio):
    """
    Calcula el valor RMS de una señal.

    El RMS da una idea del nivel promedio de amplitud.
    """
    return np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0


def calcular_clipping_ratio(audio, threshold=0.98):
    """
    Estima el porcentaje de muestras cercanas a saturación.

    Si el audio tiene muchas muestras cerca de 1 o -1,
    puede haber clipping.
    """
    if len(audio) == 0:
        return 1.0
    return np.mean(np.abs(audio) >= threshold)


def descriptor_audio_desde_lsf(lsf):
    """
    Construye un descriptor compacto del audio a partir de sus LSF.

    Estrategia:
    - se calcula la media de cada dimensión LSF,
    - se calcula la desviación estándar de cada dimensión LSF,
    - se concatenan ambos vectores.

    Este descriptor representa de forma resumida la “forma”
    general del audio dentro del espacio LSF.
    """
    mu = np.mean(lsf, axis=0)
    sigma = np.std(lsf, axis=0)
    return np.concatenate([mu, sigma])


def distancia_euclidiana(x, y):
    """
    Calcula distancia euclidiana entre dos vectores.

    Aquí se usa para comparar el descriptor de cada audio
    contra el descriptor promedio del conjunto.
    """
    return np.linalg.norm(x - y)


def extraer_metricas_audio(resultado, min_frames_validas=8):
    """
    Calcula métricas técnicas que permiten evaluar la calidad de un audio.

    Métricas calculadas:
    - duración total,
    - duración útil tras recorte,
    - RMS,
    - clipping,
    - energía,
    - ZCR,
    - número de tramas válidas,
    - porcentaje de tramas válidas,
    - error medio LPC,
    - variación entre LSF consecutivos.

    Además define una bandera llamada 'apto_tecnico', que resume
    si el audio pasa criterios básicos mínimos para ser candidato
    a entrenamiento.
    """
    audio_original = resultado["audio_original"]
    audio_recortado = resultado["audio_recortado"]
    energia = resultado["energia"]
    zcr = resultado["zcr"]
    lsf = resultado["lsf"]
    errors = resultado["errors"]
    fs = resultado["fs"]

    duracion_total_s = len(audio_original) / fs
    duracion_util_s = len(audio_recortado) / fs

    rms_original = calcular_rms(audio_original)
    rms_util = calcular_rms(audio_recortado)
    clipping_ratio = calcular_clipping_ratio(audio_original)

    energia_media = float(np.mean(energia)) if len(energia) > 0 else 0.0
    energia_std = float(np.std(energia)) if len(energia) > 0 else 0.0
    zcr_media = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    zcr_std = float(np.std(zcr)) if len(zcr) > 0 else 0.0

    num_frames_total = resultado["num_frames_total"]
    num_frames_validas = resultado["num_frames_validas"]

    porcentaje_frames_validas = (
        num_frames_validas / num_frames_total if num_frames_total > 0 else 0.0
    )

    error_lpc_medio = float(np.mean(errors)) if len(errors) > 0 else np.inf
    error_lpc_std = float(np.std(errors)) if len(errors) > 0 else np.inf

    # Si hay al menos dos vectores LSF, medimos qué tanto cambian entre sí.
    if len(lsf) >= 2:
        diffs = np.diff(lsf, axis=0)
        lsf_jump_mean = float(np.mean(np.linalg.norm(diffs, axis=1)))
        lsf_jump_std = float(np.std(np.linalg.norm(diffs, axis=1)))
    else:
        lsf_jump_mean = np.inf
        lsf_jump_std = np.inf

    # Descriptor global del audio en el espacio LSF.
    descriptor = descriptor_audio_desde_lsf(lsf)

    metricas = {
        "duracion_total_s": duracion_total_s,
        "duracion_util_s": duracion_util_s,
        "rms_original": rms_original,
        "rms_util": rms_util,
        "clipping_ratio": clipping_ratio,
        "energia_media": energia_media,
        "energia_std": energia_std,
        "zcr_media": zcr_media,
        "zcr_std": zcr_std,
        "num_frames_total": num_frames_total,
        "num_frames_validas": num_frames_validas,
        "porcentaje_frames_validas": porcentaje_frames_validas,
        "error_lpc_medio": error_lpc_medio,
        "error_lpc_std": error_lpc_std,
        "lsf_jump_mean": lsf_jump_mean,
        "lsf_jump_std": lsf_jump_std,
        "descriptor": descriptor,

        # Criterios mínimos para considerar el audio usable.
        "apto_tecnico": (
            num_frames_validas >= min_frames_validas and
            porcentaje_frames_validas >= 0.70 and
            duracion_util_s >= 0.20 and
            duracion_util_s <= 1.50 and
            rms_util >= 0.01 and
            clipping_ratio <= 0.02
        )
    }

    return metricas


# =========================================================
# CLASIFICACIÓN DE AUDIOS EN UNA SOLA CARPETA
# =========================================================

def listar_wavs(ruta_carpeta):
    """
    Devuelve una lista ordenada con todos los archivos WAV de una carpeta.
    """
    archivos = []
    for nombre in os.listdir(ruta_carpeta):
        if nombre.lower().endswith(".wav"):
            archivos.append(os.path.join(ruta_carpeta, nombre))
    archivos.sort()
    return archivos


def construir_motivo_tecnico(m):
    """
    Construye una explicación textual de por qué un audio
    no fue aceptado técnicamente.
    """
    motivos = []

    if m["num_frames_validas"] < 8:
        motivos.append("muy pocas tramas válidas")
    if m["porcentaje_frames_validas"] < 0.70:
        motivos.append("bajo porcentaje de tramas válidas")
    if m["duracion_util_s"] < 0.20:
        motivos.append("muy corto")
    if m["duracion_util_s"] > 1.50:
        motivos.append("muy largo")
    if m["rms_util"] < 0.01:
        motivos.append("nivel muy bajo")
    if m["clipping_ratio"] > 0.02:
        motivos.append("posible saturación/clipping")

    if len(motivos) == 0:
        motivos.append("falló criterio técnico")

    return ", ".join(motivos)


def guardar_csv(filas, ruta_csv):
    """
    Guarda en un archivo CSV la clasificación final de cada audio.
    """
    columnas = [
        "archivo",
        "estado",
        "motivo",
        "distancia_centroide",
        "z_score_distancia",
        "duracion_util_s",
        "rms_util",
        "clipping_ratio",
        "num_frames_validas",
        "porcentaje_frames_validas",
        "error_lpc_medio"
    ]

    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)

    print(f"\n[OK] CSV guardado en: {ruta_csv}")


def imprimir_resumen(filas_csv, media_dist, std_dist):
    """
    Imprime un resumen con conteos de audios buenos, revisables y malos.
    """
    total = len(filas_csv)
    buenos = sum(1 for x in filas_csv if x["estado"] == "bueno")
    revisar = sum(1 for x in filas_csv if x["estado"] == "revisar")
    malos = sum(1 for x in filas_csv if x["estado"] == "malo")

    print("\n================ RESUMEN ================")
    print(f"Total de audios : {total}")
    print(f"Buenos          : {buenos}")
    print(f"Revisar         : {revisar}")
    print(f"Malos           : {malos}")
    print(f"Media distancia : {media_dist:.6f}")
    print(f"STD distancia   : {std_dist:.6f}")
    print("========================================\n")


def graficar_distancias(filas_csv):
    """
    Grafica la distancia de cada audio al centroide del conjunto.

    Color:
    - verde: bueno
    - naranja: revisar
    - rojo: malo
    """
    archivos = []
    distancias = []
    colores = []

    for fila in filas_csv:
        if fila["distancia_centroide"] == "":
            continue

        archivos.append(fila["archivo"])
        distancias.append(float(fila["distancia_centroide"]))

        if fila["estado"] == "bueno":
            colores.append("green")
        elif fila["estado"] == "revisar":
            colores.append("orange")
        else:
            colores.append("red")

    if len(distancias) == 0:
        print("[INFO] No hay distancias para graficar.")
        return

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(distancias)), distancias, color=colores)
    plt.xticks(range(len(distancias)), archivos, rotation=90)
    plt.ylabel("Distancia al centroide")
    plt.title("Distancia de cada audio al centroide LSF")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def analizar_palabra(
    ruta_carpeta,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    orden_lpc=12,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65,
    min_frames_validas=8,
    umbral_outlier_std=2.0,
    graficar_outliers=False
):
    """
    Analiza todos los audios WAV de una carpeta correspondiente a una palabra.

    Objetivo:
    - detectar cuáles audios son técnicamente correctos,
    - identificar outliers respecto al comportamiento promedio,
    - y generar un reporte CSV.

    Clasificación final:
    - bueno
    - revisar
    - malo
    """
    if not os.path.isdir(ruta_carpeta):
        raise ValueError(f"La carpeta no existe: {ruta_carpeta}")

    palabra = os.path.basename(os.path.normpath(ruta_carpeta))
    salida_csv = f"reporte_{palabra}.csv"

    rutas = listar_wavs(ruta_carpeta)

    if len(rutas) == 0:
        raise ValueError(f"No se encontraron archivos .wav en la carpeta: {ruta_carpeta}")

    resultados = []
    descriptores_validos = []

    # Procesar cada audio.
    for i, ruta in enumerate(rutas):
        nombre = os.path.basename(ruta)
        print(f"[{i+1}/{len(rutas)}] Procesando: {nombre}")

        try:
            res = procesar_audio_a_caracteristicas(
                ruta,
                alpha=alpha,
                frame_length=frame_length,
                hop_length=hop_length,
                orden_lpc=orden_lpc,
                energy_factor=energy_factor,
                zcr_factor=zcr_factor,
                margen_ms=margen_ms,
                graficar=False
            )

            metricas = extraer_metricas_audio(res, min_frames_validas=min_frames_validas)

            item = {
                "ruta": ruta,
                "nombre": nombre,
                "status_procesamiento": "ok",
                "resultado": res,
                "metricas": metricas
            }

            resultados.append(item)

            # Solo los audios técnicamente válidos contribuyen al centroide grupal.
            if metricas["apto_tecnico"]:
                descriptores_validos.append(metricas["descriptor"])

        except Exception as e:
            item = {
                "ruta": ruta,
                "nombre": nombre,
                "status_procesamiento": "error",
                "resultado": None,
                "metricas": None,
                "motivo_error": str(e)
            }
            resultados.append(item)

    filas_csv = []

    # Si no hubo suficientes audios válidos, no se puede construir un centroide confiable.
    if len(descriptores_validos) < 2:
        print("\n[ADVERTENCIA] No hubo suficientes audios válidos para calcular centroide.")

        for item in resultados:
            if item["status_procesamiento"] != "ok":
                filas_csv.append({
                    "archivo": item["nombre"],
                    "estado": "malo",
                    "motivo": f"error procesamiento: {item['motivo_error']}",
                    "distancia_centroide": "",
                    "z_score_distancia": "",
                    "duracion_util_s": "",
                    "rms_util": "",
                    "clipping_ratio": "",
                    "num_frames_validas": "",
                    "porcentaje_frames_validas": "",
                    "error_lpc_medio": ""
                })
            else:
                m = item["metricas"]
                estado = "bueno" if m["apto_tecnico"] else "malo"
                motivo = "sin comparación grupal"
                if not m["apto_tecnico"]:
                    motivo = construir_motivo_tecnico(m)

                filas_csv.append({
                    "archivo": item["nombre"],
                    "estado": estado,
                    "motivo": motivo,
                    "distancia_centroide": "",
                    "z_score_distancia": "",
                    "duracion_util_s": round(m["duracion_util_s"], 5),
                    "rms_util": round(m["rms_util"], 6),
                    "clipping_ratio": round(m["clipping_ratio"], 6),
                    "num_frames_validas": m["num_frames_validas"],
                    "porcentaje_frames_validas": round(m["porcentaje_frames_validas"], 6),
                    "error_lpc_medio": round(m["error_lpc_medio"], 6)
                })

        guardar_csv(filas_csv, salida_csv)
        return filas_csv

    # Construcción del centroide promedio en el espacio descriptor.
    descriptores_validos = np.array(descriptores_validos)
    centroide = np.mean(descriptores_validos, axis=0)

    # Distancias de los audios válidos al centroide.
    distancias_validas = np.array([
        distancia_euclidiana(desc, centroide)
        for desc in descriptores_validos
    ])

    media_dist = np.mean(distancias_validas)
    std_dist = np.std(distancias_validas)

    # Evitar división entre cero.
    if std_dist < 1e-12:
        std_dist = 1e-12

    # Clasificación final archivo por archivo.
    for item in resultados:
        if item["status_procesamiento"] != "ok":
            filas_csv.append({
                "archivo": item["nombre"],
                "estado": "malo",
                "motivo": f"error procesamiento: {item['motivo_error']}",
                "distancia_centroide": "",
                "z_score_distancia": "",
                "duracion_util_s": "",
                "rms_util": "",
                "clipping_ratio": "",
                "num_frames_validas": "",
                "porcentaje_frames_validas": "",
                "error_lpc_medio": ""
            })
            continue

        m = item["metricas"]

        # Si no cumple los requisitos técnicos mínimos, se marca como malo.
        if not m["apto_tecnico"]:
            filas_csv.append({
                "archivo": item["nombre"],
                "estado": "malo",
                "motivo": construir_motivo_tecnico(m),
                "distancia_centroide": "",
                "z_score_distancia": "",
                "duracion_util_s": round(m["duracion_util_s"], 5),
                "rms_util": round(m["rms_util"], 6),
                "clipping_ratio": round(m["clipping_ratio"], 6),
                "num_frames_validas": m["num_frames_validas"],
                "porcentaje_frames_validas": round(m["porcentaje_frames_validas"], 6),
                "error_lpc_medio": round(m["error_lpc_medio"], 6)
            })
            continue

        # Si sí es técnicamente apto, se compara contra el centroide.
        desc = m["descriptor"]
        dist = distancia_euclidiana(desc, centroide)
        z_score = (dist - media_dist) / std_dist

        # Reglas de decisión:
        # - muy lejos del centroide => malo
        # - algo lejos => revisar
        # - cercano => bueno
        if z_score > umbral_outlier_std:
            estado = "malo"
            motivo = "outlier respecto al centroide"
        elif z_score > 1.0:
            estado = "revisar"
            motivo = "algo alejado del centroide"
        else:
            estado = "bueno"
            motivo = "apto técnico y cercano al centroide"

        filas_csv.append({
            "archivo": item["nombre"],
            "estado": estado,
            "motivo": motivo,
            "distancia_centroide": round(float(dist), 6),
            "z_score_distancia": round(float(z_score), 6),
            "duracion_util_s": round(m["duracion_util_s"], 5),
            "rms_util": round(m["rms_util"], 6),
            "clipping_ratio": round(m["clipping_ratio"], 6),
            "num_frames_validas": m["num_frames_validas"],
            "porcentaje_frames_validas": round(m["porcentaje_frames_validas"], 6),
            "error_lpc_medio": round(m["error_lpc_medio"], 6)
        })

    # Se guarda el reporte.
    guardar_csv(filas_csv, salida_csv)

    # Se imprime un resumen en consola.
    imprimir_resumen(filas_csv, media_dist, std_dist)

    # Opcionalmente se grafica la distancia al centroide.
    if graficar_outliers:
        graficar_distancias(filas_csv)

    return filas_csv


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # Carpeta de la palabra que se quiere analizar.
    palabra = "sigue"   # cambiar aquí según la palabra a evaluar

    # Ejecución del análisis completo del dataset de esa palabra.
    analizar_palabra(
        ruta_carpeta=palabra,
        alpha=0.95,
        frame_length=320,
        hop_length=128,
        orden_lpc=12,
        energy_factor=0.03,
        zcr_factor=0.08,
        margen_ms=65,
        min_frames_validas=8,
        umbral_outlier_std=2.0,
        graficar_outliers=True
    )