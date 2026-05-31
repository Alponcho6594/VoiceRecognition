import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct

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

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
        print(f"[INFO] {os.path.basename(ruta_audio)} convertido a mono.")

    audio = audio.astype(np.float32)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

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

    if num_samples < frame_length:
        padded = np.zeros(frame_length, dtype=audio.dtype)
        padded[:num_samples] = audio
        return padded[np.newaxis, :]

    num_frames = 1 + (num_samples - frame_length) // hop_length
    frames = np.zeros((num_frames, frame_length), dtype=audio.dtype)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i, :] = audio[start:end]

    return frames


def crear_ventana_hamming(N=320):
    """
    Genera una ventana de Hamming de tamaño N.
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
    """
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def calcular_zcr_por_trama(frames):
    """
    Calcula la tasa de cruces por cero (ZCR) para cada trama.
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
    Detecta el inicio y el final de la región con voz usando energía y ZCR.
    """
    energia = calcular_energia_por_trama(frames)
    zcr = calcular_zcr_por_trama(frames)

    energy_threshold = energy_factor * np.max(energia) if len(energia) > 0 else 0.0
    zcr_threshold = zcr_factor * np.max(zcr) if len(zcr) > 0 else 0.0

    voice_flags = (zcr > zcr_threshold) & (energia > energy_threshold)
    voice_idx = np.where(voice_flags)[0]

    if len(voice_idx) == 0:
        return None, None, energia, zcr, voice_flags, energy_threshold, zcr_threshold

    inicio_frame = voice_idx[0]
    fin_frame = voice_idx[-1]

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
# EXTRACCIÓN DE MFCC
# =========================================================

def hz_to_mel(hz):
    """
    Convierte frecuencia en Hz a escala Mel.
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """
    Convierte frecuencia en escala Mel a Hz.
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def crear_banco_mel(fs, n_fft=512, n_filters=26, fmin=0.0, fmax=None):
    """
    Crea un banco de filtros triangulares en escala Mel.
    """
    if fmax is None:
        fmax = fs / 2.0

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)

    for m in range(1, n_filters + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        if right > n_fft // 2:
            right = n_fft // 2

        for k in range(left, min(center, n_fft // 2 + 1)):
            filterbank[m - 1, k] = (k - left) / max(center - left, 1)

        for k in range(center, min(right, n_fft // 2 + 1)):
            filterbank[m - 1, k] = (right - k) / max(right - center, 1)

    return filterbank


def extraer_mfcc_por_trama(frames_hamming, fs, n_mfcc=13, n_fft=512, n_filters=26):
    """
    Extrae MFCC de cada trama ya enventanada.

    Flujo:
    1. FFT.
    2. Espectro de potencia.
    3. Banco de filtros Mel.
    4. Logaritmo de energías Mel.
    5. DCT.
    6. Selección de los primeros n_mfcc coeficientes.
    """
    spectrum = np.fft.rfft(frames_hamming, n=n_fft)
    power_spectrum = (1.0 / n_fft) * (np.abs(spectrum) ** 2)

    mel_bank = crear_banco_mel(fs, n_fft=n_fft, n_filters=n_filters)
    mel_energies = np.dot(power_spectrum, mel_bank.T)
    mel_energies = np.maximum(mel_energies, 1e-12)

    log_mel = np.log(mel_energies)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfcc.astype(np.float64), log_mel


# =========================================================
# PIPELINE COMPLETO DE UN AUDIO
# =========================================================

def procesar_audio_a_caracteristicas(
    ruta_audio,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    n_mfcc=13,
    n_fft=512,
    n_filters=26,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65,
    graficar=False
):
    """
    Procesa un archivo de audio completo y extrae MFCC.

    Pipeline:
    1. carga del audio,
    2. preénfasis,
    3. segmentación en tramas,
    4. ventana de Hamming,
    5. detección de inicio y fin,
    6. recorte de la palabra,
    7. extracción de MFCC.
    """
    fs, audio = cargar_audio(ruta_audio)
    nombre_archivo = os.path.basename(ruta_audio)

    audio_pre = aplicar_preenfasis(audio, alpha=alpha)

    frames = dividir_en_tramas(audio_pre, frame_length=frame_length, hop_length=hop_length)
    frames_hamming, _ = aplicar_ventana_hamming(frames)

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

    margen = int((margen_ms / 1000.0) * fs)
    start_sample = max(0, start_sample - margen)
    end_sample = min(len(audio_pre), end_sample + margen)

    audio_recortado = recortar_audio(audio_pre, start_sample, end_sample)

    frames_rec = dividir_en_tramas(audio_recortado, frame_length=frame_length, hop_length=hop_length)
    frames_rec_hamming, _ = aplicar_ventana_hamming(frames_rec)

    mfcc, log_mel = extraer_mfcc_por_trama(
        frames_rec_hamming,
        fs=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        n_filters=n_filters
    )

    if len(mfcc) == 0:
        raise ValueError(f"No se pudieron obtener MFCC en: {ruta_audio}")

    if graficar:
        t = np.arange(len(audio)) / fs
        t_rec = np.arange(len(audio_recortado)) / fs

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, audio)
        plt.axvline(start_sample / fs, color="g", linestyle="--", label="Inicio")
        plt.axvline(end_sample / fs, color="r", linestyle="--", label="Fin")
        plt.title(f"Audio original - {nombre_archivo}")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_rec, audio_recortado)
        plt.title(f"Audio recortado - {nombre_archivo}")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.imshow(mfcc.T, aspect="auto", origin="lower")
        plt.title(f"MFCC - {nombre_archivo}")
        plt.xlabel("Trama")
        plt.ylabel("Coeficiente MFCC")
        plt.colorbar(label="Valor")
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
        "mfcc": mfcc,
        "log_mel": log_mel,
        "num_frames_total": len(frames_rec),
        "num_frames_validas": len(mfcc)
    }


# =========================================================
# MÉTRICAS DE CALIDAD
# =========================================================

def calcular_rms(audio):
    """
    Calcula el valor RMS de una señal.
    """
    return np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0


def calcular_clipping_ratio(audio, threshold=0.98):
    """
    Estima el porcentaje de muestras cercanas a saturación.
    """
    if len(audio) == 0:
        return 1.0
    return np.mean(np.abs(audio) >= threshold)


def descriptor_audio_desde_mfcc(mfcc):
    """
    Construye un descriptor compacto del audio a partir de sus MFCC.

    Se concatena:
    - media de cada coeficiente MFCC,
    - desviación estándar de cada coeficiente MFCC.
    """
    mu = np.mean(mfcc, axis=0)
    sigma = np.std(mfcc, axis=0)
    return np.concatenate([mu, sigma])


def distancia_euclidiana(x, y):
    """
    Calcula distancia euclidiana entre dos vectores.
    """
    return np.linalg.norm(x - y)


def extraer_metricas_audio(resultado, min_frames_validas=8):
    """
    Calcula métricas técnicas para evaluar la calidad de un audio.
    """
    audio_original = resultado["audio_original"]
    audio_recortado = resultado["audio_recortado"]
    energia = resultado["energia"]
    zcr = resultado["zcr"]
    mfcc = resultado["mfcc"]
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

    if len(mfcc) >= 2:
        diffs = np.diff(mfcc, axis=0)
        mfcc_jump_mean = float(np.mean(np.linalg.norm(diffs, axis=1)))
        mfcc_jump_std = float(np.std(np.linalg.norm(diffs, axis=1)))
    else:
        mfcc_jump_mean = np.inf
        mfcc_jump_std = np.inf

    mfcc_mean_abs = float(np.mean(np.abs(mfcc))) if len(mfcc) > 0 else np.inf
    mfcc_std_global = float(np.std(mfcc)) if len(mfcc) > 0 else np.inf

    descriptor = descriptor_audio_desde_mfcc(mfcc)

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
        "mfcc_jump_mean": mfcc_jump_mean,
        "mfcc_jump_std": mfcc_jump_std,
        "mfcc_mean_abs": mfcc_mean_abs,
        "mfcc_std_global": mfcc_std_global,
        "descriptor": descriptor,
        "apto_tecnico": (
            num_frames_validas >= min_frames_validas and
            porcentaje_frames_validas >= 0.70 and
            duracion_util_s >= 0.20 and
            duracion_util_s <= 1.50 and
            rms_util >= 0.01 and
            clipping_ratio <= 0.02 and
            np.isfinite(mfcc_jump_mean)
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
    Construye una explicación textual de por qué un audio no fue aceptado.
    """
    motivos = []

    if m["num_frames_validas"] < 8:
        motivos.append("muy pocas tramas MFCC")
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
    if not np.isfinite(m["mfcc_jump_mean"]):
        motivos.append("MFCC no confiables")

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
        "mfcc_jump_mean",
        "mfcc_std_global"
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
    plt.title("Distancia de cada audio al centroide MFCC")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def analizar_palabra(
    ruta_carpeta,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    n_mfcc=13,
    n_fft=512,
    n_filters=26,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65,
    min_frames_validas=8,
    umbral_outlier_std=2.0,
    graficar_outliers=False
):
    """
    Analiza todos los audios WAV de una carpeta correspondiente a una palabra.

    Clasificación final:
    - bueno
    - revisar
    - malo
    """
    if not os.path.isdir(ruta_carpeta):
        raise ValueError(f"La carpeta no existe: {ruta_carpeta}")

    palabra = os.path.basename(os.path.normpath(ruta_carpeta))
    salida_csv = f"reporte_{palabra}_mfcc.csv"

    rutas = listar_wavs(ruta_carpeta)

    if len(rutas) == 0:
        raise ValueError(f"No se encontraron archivos .wav en la carpeta: {ruta_carpeta}")

    resultados = []
    descriptores_validos = []

    for i, ruta in enumerate(rutas):
        nombre = os.path.basename(ruta)
        print(f"[{i+1}/{len(rutas)}] Procesando: {nombre}")

        try:
            res = procesar_audio_a_caracteristicas(
                ruta,
                alpha=alpha,
                frame_length=frame_length,
                hop_length=hop_length,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                n_filters=n_filters,
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
                    "mfcc_jump_mean": "",
                    "mfcc_std_global": ""
                })
            else:
                m = item["metricas"]
                estado = "bueno" if m["apto_tecnico"] else "malo"
                motivo = "sin comparación grupal" if m["apto_tecnico"] else construir_motivo_tecnico(m)

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
                    "mfcc_jump_mean": round(m["mfcc_jump_mean"], 6) if np.isfinite(m["mfcc_jump_mean"]) else "",
                    "mfcc_std_global": round(m["mfcc_std_global"], 6) if np.isfinite(m["mfcc_std_global"]) else ""
                })

        guardar_csv(filas_csv, salida_csv)
        return filas_csv

    descriptores_validos = np.array(descriptores_validos)
    centroide = np.mean(descriptores_validos, axis=0)

    distancias_validas = np.array([
        distancia_euclidiana(desc, centroide)
        for desc in descriptores_validos
    ])

    media_dist = np.mean(distancias_validas)
    std_dist = np.std(distancias_validas)

    if std_dist < 1e-12:
        std_dist = 1e-12

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
                "mfcc_jump_mean": "",
                "mfcc_std_global": ""
            })
            continue

        m = item["metricas"]

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
                "mfcc_jump_mean": round(m["mfcc_jump_mean"], 6) if np.isfinite(m["mfcc_jump_mean"]) else "",
                "mfcc_std_global": round(m["mfcc_std_global"], 6) if np.isfinite(m["mfcc_std_global"]) else ""
            })
            continue

        desc = m["descriptor"]
        dist = distancia_euclidiana(desc, centroide)
        z_score = (dist - media_dist) / std_dist

        if z_score > umbral_outlier_std:
            estado = "malo"
            motivo = "outlier respecto al centroide MFCC"
        elif z_score > 1.0:
            estado = "revisar"
            motivo = "algo alejado del centroide MFCC"
        else:
            estado = "bueno"
            motivo = "apto técnico y cercano al centroide MFCC"

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
            "mfcc_jump_mean": round(m["mfcc_jump_mean"], 6) if np.isfinite(m["mfcc_jump_mean"]) else "",
            "mfcc_std_global": round(m["mfcc_std_global"], 6) if np.isfinite(m["mfcc_std_global"]) else ""
        })

    guardar_csv(filas_csv, salida_csv)
    imprimir_resumen(filas_csv, media_dist, std_dist)

    if graficar_outliers:
        graficar_distancias(filas_csv)

    return filas_csv


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    palabra = "alto"   # cambiar aquí según la palabra a evaluar

    analizar_palabra(
        ruta_carpeta=palabra,
        alpha=0.95,
        frame_length=320,
        hop_length=128,
        n_mfcc=13,
        n_fft=512,
        n_filters=26,
        energy_factor=0.03,
        zcr_factor=0.08,
        margen_ms=65,
        min_frames_validas=8,
        umbral_outlier_std=2.0,
        graficar_outliers=True
    )
