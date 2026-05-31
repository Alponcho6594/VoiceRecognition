import os
import glob
import json
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct

# =========================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# =========================================================

PALABRAS = [
    "alto", "busca", "carga", "trailer", "arranca"
]

# MFCC
N_MFCC = 13
N_FFT = 512
N_FILTROS_MEL = 26

# Codebook global VQ
CODEBOOK_SIZE = 256
KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-5
RANDOM_SEED = 42

# HMM discreto
NUM_ESTADOS_HMM = 5          # Puedes cambiar entre 4 y 8
NUM_SIMBOLOS = 256           # Debe coincidir con CODEBOOK_SIZE
EPSILON_B = 1e-6             # Smoothing obligatorio para B

# Audio
ALPHA_PRENFASIS = 0.95
FRAME_LENGTH = 320           # 20 ms si fs = 16 kHz
HOP_LENGTH = 128             # 8 ms si fs = 16 kHz
ENERGY_FACTOR = 0.03
ZCR_FACTOR = 0.08
MARGEN_MS = 65

# Salida
CARPETA_SALIDA = "modelos_hmm_mfcc_vq"


# =========================================================
# UTILIDADES DE JSON
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

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio = audio - np.mean(audio)
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
        "duracion_recortada_s": float(len(audio_recortado) / fs)
    }


# =========================================================
# K-MEANS GLOBAL PARA CODEBOOK 256
# =========================================================

def inicializar_centroides_kmeans(X, K, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if n < K:
        raise ValueError(
            f"No hay suficientes vectores MFCC para K={K}. "
            f"Solo hay {n} vectores. Necesitas al menos {K}."
        )

    indices = rng.choice(n, size=K, replace=False)
    return X[indices].copy()


def asignar_clusters_euclidiana(X, centroides):
    # Distancia euclidiana cuadrática usando identidad ||x-c||^2
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    c2 = np.sum(centroides ** 2, axis=1, keepdims=True).T
    dist2 = x2 + c2 - 2.0 * X @ centroides.T
    dist2 = np.maximum(dist2, 0.0)
    return np.argmin(dist2, axis=1), dist2


def entrenar_codebook_kmeans(X, K=256, max_iter=100, tol=1e-5, seed=42, verbose=True):
    centroides = inicializar_centroides_kmeans(X, K, seed=seed)
    rng = np.random.default_rng(seed + 1)

    historial = []
    distorsion_anterior = None

    for it in range(max_iter):
        asignaciones, dist2 = asignar_clusters_euclidiana(X, centroides)
        dist_min = dist2[np.arange(X.shape[0]), asignaciones]
        distorsion = float(np.mean(dist_min))

        nuevos = np.zeros_like(centroides)
        for k in range(K):
            idx = np.where(asignaciones == k)[0]
            if len(idx) == 0:
                # Reubicar cluster vacío en un punto aleatorio real.
                nuevos[k] = X[rng.integers(0, X.shape[0])]
            else:
                nuevos[k] = np.mean(X[idx], axis=0)

        if distorsion_anterior is None:
            cambio_rel = np.inf
        else:
            cambio_rel = abs(distorsion_anterior - distorsion) / max(abs(distorsion_anterior), 1e-12)

        historial.append({
            "iter": int(it),
            "distorsion_media": float(distorsion),
            "cambio_relativo": float(cambio_rel) if np.isfinite(cambio_rel) else None
        })

        if verbose:
            print(f"  iter={it:03d} | dist={distorsion:.6f} | rel={cambio_rel:.6e}")

        centroides = nuevos

        if distorsion_anterior is not None and cambio_rel < tol:
            break

        distorsion_anterior = distorsion

    # Asignaciones finales con centroides finales
    asignaciones, dist2 = asignar_clusters_euclidiana(X, centroides)
    dist_min = dist2[np.arange(X.shape[0]), asignaciones]

    return {
        "centroides": centroides,
        "asignaciones": asignaciones,
        "distorsion_media": float(np.mean(dist_min)),
        "historial": historial
    }


def mfcc_a_indices_vq(mfcc, centroides):
    indices, _ = asignar_clusters_euclidiana(mfcc, centroides)
    return indices.astype(np.int64)


# =========================================================
# RECOLECCIÓN DE DATOS MFCC
# =========================================================

def recolectar_mfcc_entrenamiento():
    mfcc_global = []
    registros = []
    fallidos = []

    print("\n====================================================")
    print("EXTRACCIÓN MFCC DE TODOS LOS AUDIOS DE ENTRENAMIENTO")
    print("====================================================")

    for palabra in PALABRAS:
        patron = os.path.join(palabra, "train", "*.wav")
        rutas = sorted(glob.glob(patron))

        print(f"\nPalabra: {palabra}")
        print(f"Archivos encontrados: {len(rutas)}")

        if len(rutas) == 0:
            fallidos.append({"palabra": palabra, "ruta": patron, "error": "sin archivos"})
            continue

        for ruta in rutas:
            try:
                datos = procesar_audio_a_mfcc(
                    ruta_audio=ruta,
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

                mfcc = datos["mfcc"]
                mfcc_global.append(mfcc)
                registros.append({
                    "palabra": palabra,
                    "ruta": ruta,
                    "mfcc": mfcc,
                    "num_frames": datos["num_frames"],
                    "duracion_recortada_s": datos["duracion_recortada_s"]
                })

                print(
                    f"[OK] {os.path.basename(ruta)} | MFCC={mfcc.shape} | "
                    f"dur={datos['duracion_recortada_s']:.3f}s"
                )

            except Exception as e:
                fallidos.append({"palabra": palabra, "ruta": ruta, "error": str(e)})
                print(f"[WARN] {ruta} -> {e}")

    if len(mfcc_global) == 0:
        raise RuntimeError("No se pudo extraer ningún MFCC. Revisa rutas y audios.")

    mfcc_global = np.vstack(mfcc_global)

    print("\nResumen MFCC global:")
    print(f"  Audios válidos      : {len(registros)}")
    print(f"  Audios fallidos     : {len(fallidos)}")
    print(f"  Matriz MFCC global  : {mfcc_global.shape}")
    print(f"  Dimensión MFCC      : {mfcc_global.shape[1]}")

    return mfcc_global, registros, fallidos


# =========================================================
# HMM POR INGENIERÍA DE CONTEOS
# =========================================================

def segmentar_linealmente(O, N):
    O = np.asarray(O, dtype=np.int64)
    T = len(O)

    if T < N:
        raise ValueError(f"La secuencia tiene T={T}, menor que N={N} estados.")

    indices = np.array_split(np.arange(T), N)
    segmentos = [O[idx] for idx in indices]
    return segmentos


def estimar_B_por_conteos(secuencias_O, N, M=256, epsilon=1e-6):
    B_counts = np.zeros((N, M), dtype=np.float64)

    for O in secuencias_O:
        segmentos = segmentar_linealmente(O, N)
        for estado, segmento in enumerate(segmentos):
            for simbolo in segmento:
                if simbolo < 0 or simbolo >= M:
                    raise ValueError(f"Símbolo fuera de rango [0,{M-1}]: {simbolo}")
                B_counts[estado, simbolo] += 1.0

    B = B_counts + epsilon
    B = B / B.sum(axis=1, keepdims=True)

    return B, B_counts


def estimar_A_por_duracion(secuencias_O, N):
    A = np.zeros((N, N), dtype=np.float64)
    duraciones_por_estado = [[] for _ in range(N)]

    for O in secuencias_O:
        segmentos = segmentar_linealmente(O, N)
        for i, segmento in enumerate(segmentos):
            duraciones_por_estado[i].append(len(segmento))

    dur_prom = np.zeros(N, dtype=np.float64)

    for i in range(N):
        dur_prom[i] = np.mean(duraciones_por_estado[i])

        if i == N - 1:
            A[i, i] = 1.0
        else:
            # Si un estado dura d frames, la probabilidad de salir en cada frame es aprox. 1/d.
            # Por lo tanto, a_ii=(d-1)/d y a_i,i+1=1/d.
            d = max(dur_prom[i], 1.0)
            A[i, i] = (d - 1.0) / d
            A[i, i + 1] = 1.0 / d

    A = A / A.sum(axis=1, keepdims=True)
    return A, dur_prom


def inicializar_pi_bakis(N):
    pi = np.zeros(N, dtype=np.float64)
    pi[0] = 1.0
    return pi


def entrenar_hmm_conteos_para_palabra(palabra, secuencias_O, N=5, M=256, epsilon=1e-6):
    if len(secuencias_O) == 0:
        raise ValueError(f"No hay secuencias O para palabra: {palabra}")

    B, B_counts = estimar_B_por_conteos(secuencias_O, N=N, M=M, epsilon=epsilon)
    A, dur_prom = estimar_A_por_duracion(secuencias_O, N=N)
    pi = inicializar_pi_bakis(N)

    # Verificaciones numéricas importantes
    if not np.allclose(A.sum(axis=1), 1.0):
        raise ValueError(f"Filas de A no suman 1 para {palabra}")
    if not np.allclose(B.sum(axis=1), 1.0):
        raise ValueError(f"Filas de B no suman 1 para {palabra}")
    if not np.allclose(pi.sum(), 1.0):
        raise ValueError(f"pi no suma 1 para {palabra}")

    modelo = {
        "palabra": palabra,
        "N": int(N),
        "M": int(M),
        "epsilon_B": float(epsilon),
        "A": A,
        "B": B,
        "B_counts": B_counts,
        "pi": pi,
        "duracion_promedio_estados": dur_prom,
        "num_secuencias_entrenamiento": int(len(secuencias_O)),
        "longitudes_secuencias": [int(len(O)) for O in secuencias_O]
    }

    return modelo


# =========================================================
# GUARDADO
# =========================================================

def guardar_codebook(modelo_codebook, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_npz = os.path.join(carpeta_salida, "codebook_mfcc_256.npz")
    ruta_json = os.path.join(carpeta_salida, "codebook_mfcc_256_historial.json")

    np.savez(
        ruta_npz,
        centroides=modelo_codebook["centroides"],
        codebook_size=CODEBOOK_SIZE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        n_filtros_mel=N_FILTROS_MEL,
        distorsion_media=modelo_codebook["distorsion_media"]
    )

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(modelo_codebook["historial"]), f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ruta_npz}")
    print(f"[GUARDADO] {ruta_json}")


def guardar_secuencias(secuencias_por_palabra, metadata_secuencias, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_npz = os.path.join(carpeta_salida, "secuencias_observacion_O.npz")
    ruta_json = os.path.join(carpeta_salida, "secuencias_observacion_O_metadata.json")

    arrays = {}
    for palabra, secuencias in secuencias_por_palabra.items():
        for i, O in enumerate(secuencias):
            arrays[f"{palabra}_{i:03d}"] = np.asarray(O, dtype=np.int64)

    np.savez(ruta_npz, **arrays)

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(metadata_secuencias), f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ruta_npz}")
    print(f"[GUARDADO] {ruta_json}")


def guardar_modelo_hmm(modelo, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)
    palabra = modelo["palabra"]

    ruta_npz = os.path.join(carpeta_salida, f"hmm_{palabra}.npz")
    ruta_json = os.path.join(carpeta_salida, f"hmm_{palabra}_resumen.json")

    np.savez(
        ruta_npz,
        palabra=palabra,
        N=modelo["N"],
        M=modelo["M"],
        epsilon_B=modelo["epsilon_B"],
        A=modelo["A"],
        B=modelo["B"],
        B_counts=modelo["B_counts"],
        pi=modelo["pi"],
        duracion_promedio_estados=modelo["duracion_promedio_estados"],
        longitudes_secuencias=np.array(modelo["longitudes_secuencias"], dtype=np.int64)
    )

    resumen = {
        "palabra": modelo["palabra"],
        "N": modelo["N"],
        "M": modelo["M"],
        "epsilon_B": modelo["epsilon_B"],
        "num_secuencias_entrenamiento": modelo["num_secuencias_entrenamiento"],
        "longitudes_secuencias": modelo["longitudes_secuencias"],
        "duracion_promedio_estados": modelo["duracion_promedio_estados"],
        "suma_filas_A": modelo["A"].sum(axis=1),
        "suma_filas_B": modelo["B"].sum(axis=1),
        "suma_pi": float(modelo["pi"].sum()),
        "A": modelo["A"]
    }

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(resumen), f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ruta_npz}")
    print(f"[GUARDADO] {ruta_json}")


# =========================================================
# PIPELINE COMPLETO DE ENTRENAMIENTO
# =========================================================

def construir_secuencias_observacion(registros, centroides):
    secuencias_por_palabra = {palabra: [] for palabra in PALABRAS}
    metadata = []

    print("\n====================================================")
    print("CUANTIZACIÓN VECTORIAL: MFCC -> ÍNDICES 0..255")
    print("====================================================")

    for reg in registros:
        palabra = reg["palabra"]
        ruta = reg["ruta"]
        mfcc = reg["mfcc"]

        O = mfcc_a_indices_vq(mfcc, centroides)
        secuencias_por_palabra[palabra].append(O)

        metadata.append({
            "palabra": palabra,
            "ruta": ruta,
            "num_frames_mfcc": int(mfcc.shape[0]),
            "longitud_O": int(len(O)),
            "min_O": int(np.min(O)),
            "max_O": int(np.max(O)),
            "primeros_20_indices": O[:20].tolist()
        })

        print(
            f"[OK] {os.path.basename(ruta)} | palabra={palabra} | "
            f"O_len={len(O)} | min={np.min(O)} | max={np.max(O)} | O[:10]={O[:10].tolist()}"
        )

    return secuencias_por_palabra, metadata


def entrenar_todos_los_hmm(secuencias_por_palabra):
    modelos_hmm = {}

    print("\n====================================================")
    print("ENTRENAMIENTO HMM POR INGENIERÍA DE CONTEOS")
    print("====================================================")

    for palabra in PALABRAS:
        secuencias = secuencias_por_palabra.get(palabra, [])

        try:
            modelo = entrenar_hmm_conteos_para_palabra(
                palabra=palabra,
                secuencias_O=secuencias,
                N=NUM_ESTADOS_HMM,
                M=NUM_SIMBOLOS,
                epsilon=EPSILON_B
            )
            modelos_hmm[palabra] = modelo

            print(f"\n[OK] HMM entrenado: {palabra}")
            print(f"  Secuencias      : {len(secuencias)}")
            print(f"  A shape         : {modelo['A'].shape}")
            print(f"  B shape         : {modelo['B'].shape}")
            print(f"  pi shape        : {modelo['pi'].shape}")
            print(f"  sum(A filas)    : {np.round(modelo['A'].sum(axis=1), 6)}")
            print(f"  sum(B filas)    : {np.round(modelo['B'].sum(axis=1), 6)}")
            print(f"  A:\n{np.round(modelo['A'], 4)}")

        except Exception as e:
            print(f"\n[ERROR] No se pudo entrenar HMM para '{palabra}': {e}")

    return modelos_hmm


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("====================================================")
    print("ENTRENAMIENTO MFCC + VQ 256 + HMM BAKIS")
    print("====================================================")
    print(f"Palabras             : {PALABRAS}")
    print(f"MFCC                 : {N_MFCC}")
    print(f"Codebook global      : {CODEBOOK_SIZE}")
    print(f"Estados HMM          : {NUM_ESTADOS_HMM}")
    print(f"Símbolos HMM         : {NUM_SIMBOLOS}")
    print(f"Salida               : {CARPETA_SALIDA}")
    print("====================================================")

    # 1. Extraer MFCC de todos los audios de train.
    mfcc_global, registros, fallidos = recolectar_mfcc_entrenamiento()

    # Guardar fallidos, si existen.
    with open(os.path.join(CARPETA_SALIDA, "audios_fallidos.json"), "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(fallidos), f, indent=2, ensure_ascii=False)

    # 2. Entrenar un único codebook global de 256 centroides.
    print("\n====================================================")
    print("ENTRENANDO CODEBOOK GLOBAL MFCC CON K-MEANS")
    print("====================================================")
    print(f"Matriz de entrenamiento: {mfcc_global.shape}")
    print(f"K objetivo             : {CODEBOOK_SIZE}")

    modelo_codebook = entrenar_codebook_kmeans(
        X=mfcc_global,
        K=CODEBOOK_SIZE,
        max_iter=KMEANS_MAX_ITER,
        tol=KMEANS_TOL,
        seed=RANDOM_SEED,
        verbose=True
    )

    print(f"\n[OK] Codebook entrenado: {modelo_codebook['centroides'].shape}")
    print(f"Distorsión media final: {modelo_codebook['distorsion_media']:.6f}")
    guardar_codebook(modelo_codebook, CARPETA_SALIDA)

    # 3. Convertir cada audio a secuencia discreta O.
    secuencias_por_palabra, metadata_secuencias = construir_secuencias_observacion(
        registros=registros,
        centroides=modelo_codebook["centroides"]
    )
    guardar_secuencias(secuencias_por_palabra, metadata_secuencias, CARPETA_SALIDA)

    # 4. Entrenar un HMM por palabra usando conteos directos.
    modelos_hmm = entrenar_todos_los_hmm(secuencias_por_palabra)

    carpeta_hmm = os.path.join(CARPETA_SALIDA, "hmm")
    os.makedirs(carpeta_hmm, exist_ok=True)

    for palabra, modelo in modelos_hmm.items():
        guardar_modelo_hmm(modelo, carpeta_hmm)

    # 5. Resumen general.
    resumen_general = {
        "palabras": PALABRAS,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "n_filtros_mel": N_FILTROS_MEL,
        "codebook_size": CODEBOOK_SIZE,
        "num_estados_hmm": NUM_ESTADOS_HMM,
        "num_simbolos": NUM_SIMBOLOS,
        "epsilon_B": EPSILON_B,
        "audios_validos": len(registros),
        "audios_fallidos": len(fallidos),
        "modelos_hmm_entrenados": list(modelos_hmm.keys()),
        "carpeta_salida": CARPETA_SALIDA
    }

    with open(os.path.join(CARPETA_SALIDA, "resumen_entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_serializable(resumen_general), f, indent=2, ensure_ascii=False)

    print("\n====================================================")
    print("FIN DEL ENTRENAMIENTO")
    print("====================================================")
    print(f"Modelos entrenados: {list(modelos_hmm.keys())}")
    print(f"Archivos guardados en: {CARPETA_SALIDA}")
    print("\nSiguiente paso: implementar Forward en logaritmos usando A, B y pi.")


if __name__ == "__main__":
    main()
