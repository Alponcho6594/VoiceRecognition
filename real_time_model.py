import os
import numpy as np
import sounddevice as sd
from scipy.fftpack import dct

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

PALABRAS = [
    "alto", "busca", "camion"
]

# Deben coincidir con entrenamiento/test
N_MFCC = 13
N_FFT = 512
N_FILTROS_MEL = 26
CODEBOOK_SIZE = 256

# Audio
FS_GRABACION = 16000
DURACION_S = 1.0
ALPHA_PRENFASIS = 0.95
FRAME_LENGTH = 320
HOP_LENGTH = 128
ENERGY_FACTOR = 0.03
ZCR_FACTOR = 0.08
MARGEN_MS = 65

# Modelos guardados por entrenamiento
CARPETA_MODELOS = "modelos_hmm_mfcc_vq"
CARPETA_HMM = os.path.join(CARPETA_MODELOS, "hmm")
RUTA_CODEBOOK = os.path.join(CARPETA_MODELOS, "codebook_mfcc_256.npz")

LOG_ZERO = -1e300


# =========================================================
# AUDIO EN MEMORIA
# =========================================================

def grabar_audio_memoria(fs=16000, duracion_s=1.0):
    """
    Graba audio desde micrófono y lo devuelve en memoria.
    No guarda ningún archivo .wav.
    """
    print("\n====================================================")
    print("GRABACIÓN EN VIVO")
    print("====================================================")
    print(f"Frecuencia : {fs} Hz")
    print(f"Duración   : {duracion_s:.2f} s")
    print("Di la palabra cuando veas: Grabando...")
    input("Presiona ENTER para comenzar...")

    print("Grabando...")
    audio = sd.rec(
        int(duracion_s * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("Grabación terminada.")

    audio = np.squeeze(audio).astype(np.float32)
    return fs, audio


def normalizar_audio(audio):
    """
    Normaliza amplitud y elimina offset DC.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 0:
        audio = audio / max_val

    if len(audio) > 0:
        audio = audio - np.mean(audio)

    return audio.astype(np.float32)


# =========================================================
# PREPROCESAMIENTO
# =========================================================

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
        if right > n_fft // 2:
            right = n_fft // 2

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

    return mfcc.astype(np.float64), log_mel


def procesar_audio_array_a_mfcc(
    audio,
    fs,
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
    """
    Procesa audio en memoria hasta MFCC.
    Esta función reemplaza cargar_audio(ruta_audio) porque aquí no hay archivo.
    """
    audio = normalizar_audio(audio)
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
        raise ValueError("No se detectó voz. Habla más fuerte o reduce ruido ambiental.")

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
        raise ValueError("No se pudieron obtener MFCC del audio grabado.")

    return {
        "fs": fs,
        "audio_original": audio,
        "audio_recortado": audio_recortado,
        "mfcc": mfcc,
        "log_mel": log_mel,
        "num_frames": int(mfcc.shape[0]),
        "duracion_original_s": float(len(audio) / fs),
        "duracion_recortada_s": float(len(audio_recortado) / fs),
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
            "Primero ejecuta tu script de entrenamiento."
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
                "Primero ejecuta tu script de entrenamiento."
            )

        data = np.load(ruta_modelo, allow_pickle=True)

        A = data["A"].astype(np.float64)
        B = data["B"].astype(np.float64)
        pi = data["pi"].astype(np.float64)
        N = int(data["N"])
        M = int(data["M"])

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
    alpha[0, :] = logpi + logB[:, O[0]]

    for t in range(1, T):
        obs = O[t]
        for j in range(N):
            valores = alpha[t - 1, :] + logA[:, j]
            alpha[t, j] = logB[j, obs] + logsumexp(valores)

    return float(logsumexp(alpha[T - 1, :]))


def clasificar_secuencia(O, modelos_hmm):
    scores = {}
    for palabra, modelo in modelos_hmm.items():
        scores[palabra] = forward_log(O, modelo)

    prediccion = max(scores, key=scores.get)
    return prediccion, scores


def reconocer_audio_en_memoria(audio, fs, centroides, modelos_hmm):
    datos = procesar_audio_array_a_mfcc(
        audio=audio,
        fs=fs,
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
# LOOP EN VIVO
# =========================================================

def imprimir_resultado(prediccion, scores, O, datos):
    print("\n====================================================")
    print("RESULTADO")
    print("====================================================")
    print(f"Palabra reconocida       : {prediccion.upper()}")
    print(f"Duración recortada       : {datos['duracion_recortada_s']:.4f} s")
    print(f"MFCC shape               : {datos['mfcc'].shape}")
    print(f"Longitud O               : {len(O)}")
    print(f"Primeros 20 índices O    : {O[:20].tolist()}")

    print("\nLog-likelihoods:")
    for palabra, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marca = "<-- ganador" if palabra == prediccion else ""
        print(f"  {palabra:>8s}: {score:12.4f} {marca}")

    ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ordenados) >= 2:
        margen = ordenados[0][1] - ordenados[1][1]
        print(f"\nMargen ganador vs segundo: {margen:.4f}")


def main():
    print("====================================================")
    print("RECONOCIMIENTO EN VIVO: MICRÓFONO + MFCC + VQ + HMM")
    print("====================================================")
    print(f"Palabras        : {PALABRAS}")
    print(f"Modelos         : {CARPETA_MODELOS}")
    print(f"Codebook        : {RUTA_CODEBOOK}")
    print(f"HMM             : {CARPETA_HMM}")
    print("====================================================")

    centroides = cargar_codebook(RUTA_CODEBOOK)
    modelos_hmm = cargar_modelos_hmm(CARPETA_HMM, PALABRAS)

    while True:
        try:
            fs, audio = grabar_audio_memoria(fs=FS_GRABACION, duracion_s=DURACION_S)
            prediccion, scores, O, datos = reconocer_audio_en_memoria(audio, fs, centroides, modelos_hmm)
            imprimir_resultado(prediccion, scores, O, datos)

        except Exception as e:
            print(f"\n[ERROR] {e}")

        opcion = input("\n¿Grabar otra palabra? [s/n]: ").strip().lower()
        if opcion not in ("s", "si", "sí", "y", "yes"):
            print("Fin del reconocimiento en vivo.")
            break


if __name__ == "__main__":
    main()
