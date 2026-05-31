import os
import glob
import csv
import json
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct

# =========================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# =========================================================

# IMPORTANTE:
# Esta lista debe coincidir exactamente con las palabras usadas en entrenamiento.
# Si tu proyecto es de 5 palabras, la matriz de confusión será 5x5.
PALABRAS = [
    "alto", "busca", "carga", "trailer", "arranca"
]

# Deben coincidir con el entrenamiento
N_MFCC = 13
N_FFT = 512
N_FILTROS_MEL = 26
CODEBOOK_SIZE = 256
NUM_SIMBOLOS = 256

# Audio
ALPHA_PRENFASIS = 0.95
FRAME_LENGTH = 320
HOP_LENGTH = 128
ENERGY_FACTOR = 0.03
ZCR_FACTOR = 0.08
MARGEN_MS = 65

# Modelos guardados por el entrenamiento
CARPETA_MODELOS = "modelos_hmm_mfcc_vq"
CARPETA_HMM = os.path.join(CARPETA_MODELOS, "hmm")
RUTA_CODEBOOK = os.path.join(CARPETA_MODELOS, "codebook_mfcc_256.npz")

# Salidas de prueba
ARCHIVO_MATRIZ = "matriz_confusion_mfcc_hmm_forward.csv"
ARCHIVO_REPORTE = "reporte_test_mfcc_hmm_forward.csv"
ARCHIVO_RESUMEN = "resumen_test_mfcc_hmm_forward.json"

# Entregables de verificación
ARCHIVO_VERIFICACION_A = "verificacion_diagonalidad_A.csv"
ARCHIVO_VERIFICACION_B = "verificacion_sparsity_B_estado1.csv"
ARCHIVO_VERIFICACION_JSON = "verificacion_modelos_hmm.json"

# Forward log
LOG_ZERO = -1e300


# =========================================================
# UTILIDADES
# =========================================================

def convertir_a_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convertir_a_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convertir_a_json_serializable(x) for x in obj]
    return obj


# =========================================================
# CARGA DE AUDIO Y PREPROCESAMIENTO
# =========================================================

def cargar_audio(ruta_audio):
    fs, audio = wavfile.read(ruta_audio)

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
        print(f"[INFO] Audio convertido a mono: {ruta_audio}")

    audio = audio.astype(np.float32)

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 0:
        audio = audio / max_val

    audio = audio - np.mean(audio) if len(audio) > 0 else audio
    return fs, audio


def aplicar_preenfasis(audio, alpha=0.95):
    if len(audio) == 0:
        return audio

    audio_pre = np.empty_like(audio)
    audio_pre[0] = audio[0]
    audio_pre[1:] = audio[1:] - alpha * audio[:-1]
    return audio_pre


def dividir_en_tramas(audio, frame_length=320, hop_length=128):
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
    n = np.arange(N)
    w = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))
    return w.astype(np.float32)


def aplicar_ventana_hamming(frames):
    ventana = crear_ventana_hamming(frames.shape[1])
    return frames * ventana, ventana


def calcular_energia_por_trama(frames):
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def calcular_zcr_por_trama(frames):
    zcr = np.zeros(frames.shape[0], dtype=np.float32)

    for i, frame in enumerate(frames):
        signos = np.sign(frame)
        signos[signos == 0] = 1
        crossings = np.sum(np.abs(np.diff(signos))) / 2
        zcr[i] = crossings / len(frame)

    return zcr


def detectar_inicio_fin(frames, hop_length, frame_length, energy_factor=0.03, zcr_factor=0.08):
    energia = calcular_energia_por_trama(frames)
    zcr = calcular_zcr_por_trama(frames)

    energy_threshold = energy_factor * np.max(energia) if len(energia) > 0 else 0.0
    zcr_threshold = zcr_factor * np.max(zcr) if len(zcr) > 0 else 0.0

    voice_flags = (energia > energy_threshold) & (zcr > zcr_threshold)
    voice_idx = np.where(voice_flags)[0]

    if len(voice_idx) == 0:
        return None, None, energia, zcr, voice_flags, energy_threshold, zcr_threshold

    inicio_frame = voice_idx[0]
    fin_frame = voice_idx[-1]

    start_sample = inicio_frame * hop_length
    end_sample = fin_frame * hop_length + frame_length

    return start_sample, end_sample, energia, zcr, voice_flags, energy_threshold, zcr_threshold


def recortar_audio(audio, start_sample, end_sample):
    if start_sample is None or end_sample is None:
        return None
    end_sample = min(end_sample, len(audio))
    return audio[start_sample:end_sample]


# =========================================================
# MFCC
# =========================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def crear_banco_mel(fs, n_fft=512, n_filters=26, fmin=0.0, fmax=None):
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

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft // 2)

        for k in range(left, min(center, n_fft // 2 + 1)):
            filterbank[m - 1, k] = (k - left) / max(center - left, 1)

        for k in range(center, min(right, n_fft // 2 + 1)):
            filterbank[m - 1, k] = (right - k) / max(right - center, 1)

    return filterbank


def extraer_mfcc_por_trama(frames_hamming, fs, n_mfcc=13, n_fft=512, n_filters=26):
    spectrum = np.fft.rfft(frames_hamming, n=n_fft)
    power_spectrum = (1.0 / n_fft) * (np.abs(spectrum) ** 2)

    mel_bank = crear_banco_mel(fs, n_fft=n_fft, n_filters=n_filters)
    mel_energies = np.dot(power_spectrum, mel_bank.T)
    mel_energies = np.maximum(mel_energies, 1e-12)

    log_mel = np.log(mel_energies)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfcc.astype(np.float64)


def procesar_audio_a_mfcc(
    ruta_audio,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    n_mfcc=13,
    n_fft=512,
    n_filters=26,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65
):
    fs, audio = cargar_audio(ruta_audio)
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

    mfcc = extraer_mfcc_por_trama(
        frames_rec_hamming,
        fs=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        n_filters=n_filters
    )

    if mfcc.shape[0] == 0:
        raise ValueError(f"No se pudieron obtener MFCC en: {ruta_audio}")

    return {
        "ruta": ruta_audio,
        "fs": fs,
        "mfcc": mfcc,
        "num_frames": int(mfcc.shape[0]),
        "duracion_original_s": len(audio) / fs,
        "duracion_recortada_s": len(audio_recortado) / fs,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "energy_threshold": float(energy_threshold),
        "zcr_threshold": float(zcr_threshold)
    }


# =========================================================
# VQ: MFCC -> ÍNDICES 0..255
# =========================================================

def cargar_codebook(ruta_codebook=RUTA_CODEBOOK):
    if not os.path.exists(ruta_codebook):
        raise FileNotFoundError(
            f"No existe el codebook: {ruta_codebook}\n"
            "Primero ejecuta el archivo de entrenamiento."
        )

    data = np.load(ruta_codebook, allow_pickle=True)
    centroides = data["centroides"].astype(np.float64)

    if centroides.shape[0] != CODEBOOK_SIZE:
        raise ValueError(f"El codebook debe tener 256 centroides, tiene: {centroides.shape[0]}")

    print(f"[MODELO] Codebook cargado: {ruta_codebook} | shape={centroides.shape}")
    return centroides


def mfcc_a_indices_vq(mfcc, centroides):
    mfcc = np.asarray(mfcc, dtype=np.float64)
    centroides = np.asarray(centroides, dtype=np.float64)

    distancias = np.sum((mfcc[:, None, :] - centroides[None, :, :]) ** 2, axis=2)
    O = np.argmin(distancias, axis=1).astype(np.int64)

    if np.any(O < 0) or np.any(O >= CODEBOOK_SIZE):
        raise ValueError("La secuencia O contiene índices fuera del rango 0..255")

    return O


# =========================================================
# CARGA DE HMMs
# =========================================================

def cargar_modelos_hmm(carpeta_hmm=CARPETA_HMM, palabras=PALABRAS):
    modelos = {}

    for palabra in palabras:
        ruta_modelo = os.path.join(carpeta_hmm, f"hmm_{palabra}.npz")

        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(
                f"No existe el HMM: {ruta_modelo}\n"
                "Revisa que entrenaste exactamente las mismas palabras."
            )

        data = np.load(ruta_modelo, allow_pickle=True)

        A = data["A"].astype(np.float64)
        B = data["B"].astype(np.float64)
        pi = data["pi"].astype(np.float64)
        N = int(data["N"])
        M = int(data["M"])

        if A.shape != (N, N):
            raise ValueError(f"A inválida en {palabra}: {A.shape}")
        if B.shape != (N, M):
            raise ValueError(f"B inválida en {palabra}: {B.shape}")
        if pi.shape != (N,):
            raise ValueError(f"pi inválida en {palabra}: {pi.shape}")
        if M != NUM_SIMBOLOS:
            raise ValueError(f"M debe ser 256 en {palabra}, pero es {M}")
        if not np.allclose(A.sum(axis=1), 1.0):
            raise ValueError(f"Las filas de A no suman 1 en {palabra}")
        if not np.allclose(B.sum(axis=1), 1.0):
            raise ValueError(f"Las filas de B no suman 1 en {palabra}")
        if not np.allclose(pi.sum(), 1.0):
            raise ValueError(f"pi no suma 1 en {palabra}")

        modelos[palabra] = {
            "palabra": palabra,
            "N": N,
            "M": M,
            "A": A,
            "B": B,
            "pi": pi,
            "ruta_modelo": ruta_modelo
        }

        print(f"[MODELO] HMM cargado: {palabra:>8s} | A={A.shape} | B={B.shape} | pi={pi.shape}")

    return modelos


# =========================================================
# FORWARD EN LOGARITMOS
# =========================================================

def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values)

    if vmax <= LOG_ZERO / 2:
        return LOG_ZERO

    return vmax + np.log(np.sum(np.exp(values - vmax)))


def forward_log(O, modelo):
    """
    Calcula log P(O | lambda) usando Forward en espacio logarítmico.
    La palabra reconocida es la del modelo con mayor log-likelihood.
    """
    O = np.asarray(O, dtype=np.int64)

    A = modelo["A"]
    B = modelo["B"]
    pi = modelo["pi"]
    N = modelo["N"]
    M = modelo["M"]
    T = len(O)

    if T == 0:
        raise ValueError("La secuencia O está vacía.")
    if np.any(O < 0) or np.any(O >= M):
        raise ValueError(f"O contiene símbolos fuera del rango 0..{M - 1}")

    logA = np.where(A > 0, np.log(A), LOG_ZERO)
    logB = np.where(B > 0, np.log(B), LOG_ZERO)
    logpi = np.where(pi > 0, np.log(pi), LOG_ZERO)

    alpha = np.full((T, N), LOG_ZERO, dtype=np.float64)

    # Inicialización
    alpha[0, :] = logpi + logB[:, O[0]]

    # Recursión
    for t in range(1, T):
        obs = O[t]
        for j in range(N):
            valores = alpha[t - 1, :] + logA[:, j]
            alpha[t, j] = logB[j, obs] + logsumexp(valores)

    # Terminación
    return float(logsumexp(alpha[T - 1, :]))


def clasificar_secuencia(O, modelos_hmm):
    scores = {}

    for palabra, modelo in modelos_hmm.items():
        scores[palabra] = forward_log(O, modelo)

    prediccion = max(scores, key=scores.get)
    return prediccion, scores


def reconocer_audio(ruta_audio, centroides, modelos_hmm):
    datos = procesar_audio_a_mfcc(
        ruta_audio=ruta_audio,
        alpha=ALPHA_PRENFASIS,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        n_filters=N_FILTROS_MEL,
        energy_factor=ENERGY_FACTOR,
        zcr_factor=ZCR_FACTOR,
        margen_ms=MARGEN_MS
    )

    O = mfcc_a_indices_vq(datos["mfcc"], centroides)
    prediccion, scores = clasificar_secuencia(O, modelos_hmm)

    return prediccion, scores, O, datos


# =========================================================
# ENTREGABLES DE VERIFICACIÓN
# =========================================================

def verificar_diagonalidad_A(modelos_hmm):
    """
    Entregable:
    Demuestra que A es Bakis / left-to-right:
    solo permite a_ii y a_i,i+1.
    """
    filas_csv = []
    resumen = {}

    print("\n====================================================")
    print("VERIFICACIÓN: DIAGONALIDAD / BAKIS EN A")
    print("====================================================")

    for palabra, modelo in modelos_hmm.items():
        A = modelo["A"]
        N = modelo["N"]

        mascara_permitida = np.zeros_like(A, dtype=bool)
        for i in range(N):
            mascara_permitida[i, i] = True
            if i + 1 < N:
                mascara_permitida[i, i + 1] = True

        masa_permitida = float(np.sum(A[mascara_permitida]))
        masa_prohibida = float(np.sum(A[~mascara_permitida]))
        cumple_bakis = np.isclose(masa_prohibida, 0.0)

        resumen[palabra] = {
            "A": A,
            "masa_permitida_diagonal_superdiagonal": masa_permitida,
            "masa_prohibida_fuera_de_bakis": masa_prohibida,
            "cumple_bakis": bool(cumple_bakis)
        }

        print(f"\nPalabra: {palabra}")
        print(np.round(A, 6))
        print(f"Masa en diagonal/superdiagonal: {masa_permitida:.6f}")
        print(f"Masa fuera de Bakis           : {masa_prohibida:.12f}")
        print(f"Cumple Bakis                  : {cumple_bakis}")

        for i in range(N):
            fila = {"palabra": palabra, "estado": i + 1}
            for j in range(N):
                fila[f"A_{i+1}_{j+1}"] = A[i, j]
            filas_csv.append(fila)

    if filas_csv:
        columnas = ["palabra", "estado"] + [
            f"A_{i+1}_{j+1}"
            for i in range(next(iter(modelos_hmm.values()))["N"])
            for j in range(next(iter(modelos_hmm.values()))["N"])
        ]

        # Para evitar muchas columnas vacías por fila, guardamos una fila por estado con todas las columnas.
        with open(ARCHIVO_VERIFICACION_A, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            for fila in filas_csv:
                writer.writerow(fila)

    print(f"\n[GUARDADO] {ARCHIVO_VERIFICACION_A}")
    return resumen


def verificar_sparsity_B_estado1(modelos_hmm, top_k=10, umbral_casi_cero=1e-5):
    """
    Entregable:
    Para el Estado 1, muestra que B tiene picos en pocos índices
    y probabilidades casi cero en la mayoría de los 256 símbolos.
    """
    filas_csv = []
    resumen = {}

    print("\n====================================================")
    print("VERIFICACIÓN: SPARSITY EN B, ESTADO 1")
    print("====================================================")

    for palabra, modelo in modelos_hmm.items():
        B = modelo["B"]
        b_estado1 = B[0, :]  # Estado 1, índices 0..255

        indices_ordenados = np.argsort(b_estado1)[::-1]
        top_indices = indices_ordenados[:top_k]

        num_casi_cero = int(np.sum(b_estado1 <= umbral_casi_cero))
        porcentaje_casi_cero = 100.0 * num_casi_cero / len(b_estado1)

        resumen[palabra] = {
            "top_indices_estado1": top_indices.tolist(),
            "top_probabilidades_estado1": b_estado1[top_indices].tolist(),
            "num_simbolos_casi_cero": num_casi_cero,
            "porcentaje_simbolos_casi_cero": porcentaje_casi_cero,
            "umbral_casi_cero": umbral_casi_cero
        }

        print(f"\nPalabra: {palabra}")
        print(f"Símbolos casi cero <= {umbral_casi_cero}: {num_casi_cero}/256 ({porcentaje_casi_cero:.2f}%)")
        print("Top índices con mayor probabilidad en B[Estado 1]:")
        for rank, idx in enumerate(top_indices, start=1):
            prob = float(b_estado1[idx])
            print(f"  {rank:02d}. índice={idx:3d} | prob={prob:.10f}")

            filas_csv.append({
                "palabra": palabra,
                "estado": 1,
                "rank": rank,
                "indice_vq": int(idx),
                "probabilidad": prob,
                "num_simbolos_casi_cero": num_casi_cero,
                "porcentaje_simbolos_casi_cero": porcentaje_casi_cero,
                "umbral_casi_cero": umbral_casi_cero
            })

    columnas = [
        "palabra",
        "estado",
        "rank",
        "indice_vq",
        "probabilidad",
        "num_simbolos_casi_cero",
        "porcentaje_simbolos_casi_cero",
        "umbral_casi_cero"
    ]

    with open(ARCHIVO_VERIFICACION_B, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in filas_csv:
            writer.writerow(fila)

    print(f"\n[GUARDADO] {ARCHIVO_VERIFICACION_B}")
    return resumen


def guardar_verificacion_json(resumen_A, resumen_B):
    resumen = {
        "palabras": PALABRAS,
        "nota": "Si el proyecto usa 5 palabras, la matriz de confusión correcta es 5x5. Si pidieran 10 palabras, cambiar PALABRAS a 10 clases.",
        "diagonalidad_A": resumen_A,
        "sparsity_B_estado1": resumen_B,
        "archivo_verificacion_A": ARCHIVO_VERIFICACION_A,
        "archivo_verificacion_B": ARCHIVO_VERIFICACION_B
    }

    with open(ARCHIVO_VERIFICACION_JSON, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(resumen), f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ARCHIVO_VERIFICACION_JSON}")


# =========================================================
# REPORTES Y MATRIZ DE CONFUSIÓN
# =========================================================

def crear_matriz_confusion(palabras):
    return np.zeros((len(palabras), len(palabras)), dtype=int)


def guardar_matriz_confusion_csv(matriz, palabras, ruta_csv):
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real\\pred"] + palabras)
        for i, palabra_real in enumerate(palabras):
            writer.writerow([palabra_real] + list(matriz[i]))


def guardar_reporte_csv(reporte, ruta_csv, palabras):
    columnas_base = [
        "archivo",
        "real",
        "predicha",
        "correcta",
        "num_frames_mfcc",
        "longitud_O",
        "duracion_recortada_s",
        "primeros_20_indices"
    ]

    columnas_scores = [f"loglik_{p}" for p in palabras]
    columnas = columnas_base + columnas_scores

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in reporte:
            writer.writerow(fila)


def imprimir_matriz_confusion(matriz, palabras):
    print("\n====================================================")
    print("MATRIZ DE CONFUSIÓN")
    print("====================================================")
    print("Filas = palabra real | Columnas = predicción\n")

    print("          " + "  ".join([f"{p:>8s}" for p in palabras]))
    for i, palabra_real in enumerate(palabras):
        fila = "  ".join([f"{matriz[i, j]:8d}" for j in range(len(palabras))])
        print(f"{palabra_real:>8s}  {fila}")

    print(f"\nTamaño de matriz: {len(palabras)}x{len(palabras)}")


def analizar_confusiones(matriz, palabras):
    """
    Genera un análisis básico de confusiones.
    El análisis fonético fino debe completarse manualmente según las palabras confundidas.
    """
    analisis = []

    for i, palabra_real in enumerate(palabras):
        total_real = int(np.sum(matriz[i, :]))
        correctas = int(matriz[i, i])

        for j, palabra_pred in enumerate(palabras):
            if i != j and matriz[i, j] > 0:
                analisis.append({
                    "real": palabra_real,
                    "predicha": palabra_pred,
                    "conteo": int(matriz[i, j]),
                    "total_real": total_real,
                    "comentario": (
                        "Revisar si hay similitud fonética entre estas palabras o si el codebook VQ "
                        "está asignando índices parecidos a regiones acústicas distintas."
                    )
                })

        if total_real > 0 and correctas < total_real:
            print(
                f"[ANÁLISIS] La palabra '{palabra_real}' tuvo {total_real - correctas} errores "
                f"de {total_real} muestras."
            )

    if len(analisis) == 0:
        print("[ANÁLISIS] No hubo confusiones registradas en la matriz.")
    else:
        print("\n====================================================")
        print("ANÁLISIS BÁSICO DE CONFUSIONES")
        print("====================================================")
        for item in analisis:
            print(
                f"Real='{item['real']}' -> Predicha='{item['predicha']}' | "
                f"conteo={item['conteo']}/{item['total_real']}"
            )
            print(f"  {item['comentario']}")

    return analisis


# =========================================================
# EVALUACIÓN COMPLETA
# =========================================================

def probar_modelo():
    print("====================================================")
    print("TEST MFCC + VQ 256 + HMM BAKIS + FORWARD LOG")
    print("====================================================")
    print(f"Palabras        : {PALABRAS}")
    print(f"Carpeta modelos : {CARPETA_MODELOS}")
    print(f"Codebook        : {RUTA_CODEBOOK}")
    print(f"HMM             : {CARPETA_HMM}")
    print("====================================================")

    centroides = cargar_codebook(RUTA_CODEBOOK)
    modelos_hmm = cargar_modelos_hmm(CARPETA_HMM, PALABRAS)

    # Entregables de verificación antes del test
    resumen_A = verificar_diagonalidad_A(modelos_hmm)
    resumen_B = verificar_sparsity_B_estado1(modelos_hmm)
    guardar_verificacion_json(resumen_A, resumen_B)

    matriz = crear_matriz_confusion(PALABRAS)
    reporte = []
    errores = []

    total = 0
    correctos = 0

    for i_real, palabra_real in enumerate(PALABRAS):
        rutas_test = sorted(glob.glob(os.path.join(palabra_real, "test", "*.wav")))

        print("\n----------------------------------------------------")
        print(f"Evaluando palabra real: {palabra_real}")
        print(f"Archivos test encontrados: {len(rutas_test)}")
        print("----------------------------------------------------")

        if len(rutas_test) == 0:
            errores.append({
                "palabra": palabra_real,
                "ruta": os.path.join(palabra_real, "test", "*.wav"),
                "error": "sin archivos"
            })
            continue

        for ruta_audio in rutas_test:
            try:
                prediccion, scores, O, datos = reconocer_audio(
                    ruta_audio=ruta_audio,
                    centroides=centroides,
                    modelos_hmm=modelos_hmm
                )

                j_pred = PALABRAS.index(prediccion)
                matriz[i_real, j_pred] += 1

                es_correcta = prediccion == palabra_real
                total += 1
                if es_correcta:
                    correctos += 1

                fila = {
                    "archivo": ruta_audio,
                    "real": palabra_real,
                    "predicha": prediccion,
                    "correcta": int(es_correcta),
                    "num_frames_mfcc": int(datos["num_frames"]),
                    "longitud_O": int(len(O)),
                    "duracion_recortada_s": round(float(datos["duracion_recortada_s"]), 6),
                    "primeros_20_indices": " ".join(map(str, O[:20].tolist()))
                }

                for palabra_modelo in PALABRAS:
                    fila[f"loglik_{palabra_modelo}"] = round(float(scores[palabra_modelo]), 6)

                reporte.append(fila)

                print(f"[TEST] {ruta_audio}")
                print(
                    f"       real={palabra_real} | pred={prediccion} | "
                    f"ok={es_correcta} | MFCC={datos['mfcc'].shape} | O_len={len(O)}"
                )
                print(f"       O[:20]={O[:20].tolist()}")

                scores_ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for palabra_modelo, score in scores_ordenados:
                    marca = "<-- ganador" if palabra_modelo == prediccion else ""
                    print(f"       {palabra_modelo:>8s}: {score:12.4f} {marca}")

            except Exception as e:
                errores.append({"palabra": palabra_real, "ruta": ruta_audio, "error": str(e)})
                print(f"[ERROR] {ruta_audio} -> {e}")

    accuracy = (correctos / total) * 100.0 if total > 0 else 0.0

    imprimir_matriz_confusion(matriz, PALABRAS)
    analisis_confusiones = analizar_confusiones(matriz, PALABRAS)

    print("\n====================================================")
    print(f"Accuracy total: {accuracy:.2f}% ({correctos}/{total})")
    print("====================================================")

    guardar_matriz_confusion_csv(matriz, PALABRAS, ARCHIVO_MATRIZ)
    guardar_reporte_csv(reporte, ARCHIVO_REPORTE, PALABRAS)

    resumen = {
        "accuracy": accuracy,
        "correctos": correctos,
        "total": total,
        "palabras": PALABRAS,
        "tamano_matriz_confusion": f"{len(PALABRAS)}x{len(PALABRAS)}",
        "matriz_confusion": matriz,
        "analisis_confusiones": analisis_confusiones,
        "errores": errores,
        "archivo_matriz": ARCHIVO_MATRIZ,
        "archivo_reporte": ARCHIVO_REPORTE,
        "archivo_verificacion_A": ARCHIVO_VERIFICACION_A,
        "archivo_verificacion_B": ARCHIVO_VERIFICACION_B,
        "archivo_verificacion_json": ARCHIVO_VERIFICACION_JSON,
        "carpeta_modelos": CARPETA_MODELOS,
        "ruta_codebook": RUTA_CODEBOOK,
        "carpeta_hmm": CARPETA_HMM
    }

    with open(ARCHIVO_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(resumen), f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ARCHIVO_MATRIZ}")
    print(f"[GUARDADO] {ARCHIVO_REPORTE}")
    print(f"[GUARDADO] {ARCHIVO_RESUMEN}")

    if len(errores) > 0:
        print("\n[ADVERTENCIA] Hubo errores o carpetas sin audios:")
        for err in errores:
            print(f"  - {err['ruta']} -> {err['error']}")

    return resumen


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    probar_modelo()
