import os
import glob
import csv
import numpy as np
from scipy.io import wavfile

# =========================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# =========================================================

# Lista de palabras que formarán las clases del sistema.
# Cada palabra tendrá su propio conjunto de grabaciones y su propio codebook.
PALABRAS = [
    "alto", "busca", "camion", "carga", "deja",
    "inicio", "mapa", "pausa", "ruta", "sigue"
]

# Orden del modelo LPC.
# La práctica pide usar LPC de orden 12.
ORDEN_LPC = 12

# Tamaños de codebook a probar.
# La práctica pide investigar con 16, 32 y 64 codevectors.
CODEBOOK_SIZES = [16, 32, 64]

# Coeficiente del filtro de preénfasis.
# Implementa H(z) = 1 - 0.95 z^-1
ALPHA_PRENFASIS = 0.95

# Longitud de cada trama.
# 320 muestras a 16 kHz equivalen a 20 ms.
FRAME_LENGTH = 320

# Corrimiento entre tramas.
# 128 muestras equivalen a 8 ms.
HOP_LENGTH = 128

# Factores para construir los umbrales de energía y ZCR.
# Se multiplican por el valor máximo encontrado en cada archivo.
ENERGY_FACTOR = 0.03
ZCR_FACTOR = 0.08

# Margen adicional al recorte de voz, en milisegundos.
# Se agrega un poco antes y después para no cortar bruscamente la palabra.
MARGEN_MS = 65

# Tolerancia relativa para detener el entrenamiento del LBG.
# Si la distorsión cambia muy poco, se considera que ya convergió.
TOL_REL = 1e-5

# Número máximo de iteraciones del algoritmo LBG.
MAX_ITER_LBG = 50

# Pequeña perturbación usada al dividir centroides en LBG.
DELTA_SPLIT = 0.0001


# =========================================================
# CARGA DE AUDIO Y PREPROCESAMIENTO
# =========================================================

def cargar_audio(ruta_audio):
    """
    Lee un archivo WAV y lo prepara para el procesamiento.

    Qué hace:
    1. Lee el archivo.
    2. Si tiene dos canales, lo convierte a mono.
    3. Lo convierte a float32.
    4. Normaliza la amplitud.
    5. Elimina el valor medio.
    """
    fs, audio = wavfile.read(ruta_audio)

    # Si el audio es estéreo, promedia ambos canales para trabajar en mono.
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
        print(f"[INFO] Audio convertido a mono: {ruta_audio}")

    # Convertimos a punto flotante para evitar problemas en operaciones posteriores.
    audio = audio.astype(np.float32)

    # Normalización de amplitud.
    # Esto ayuda a que distintas grabaciones tengan una escala comparable.
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Quitamos la componente DC.
    # Es decir, centramos la señal alrededor de cero.
    audio = audio - np.mean(audio)

    return fs, audio


def aplicar_preenfasis(audio, alpha=0.95):
    """
    Aplica el filtro de preénfasis.

    Fórmula:
        y[n] = x[n] - alpha * x[n-1]
    """
    audio_pre = np.empty_like(audio)

    # La primera muestra no tiene muestra anterior.
    audio_pre[0] = audio[0]

    # Aplicamos la ecuación del filtro a partir de la segunda muestra.
    audio_pre[1:] = audio[1:] - alpha * audio[:-1]

    return audio_pre


def dividir_en_tramas(audio, frame_length=320, hop_length=128):
    """
    Divide el audio en tramas o bloques.

    Si la señal es más corta que una trama completa, se rellena con ceros.
    """
    num_samples = len(audio)

    # Si el audio es muy corto, lo rellenamos para formar una sola trama.
    if num_samples < frame_length:
        padded = np.zeros(frame_length, dtype=audio.dtype)
        padded[:num_samples] = audio
        return padded[np.newaxis, :]

    # Número de tramas completas que caben con el salto definido.
    num_frames = 1 + (num_samples - frame_length) // hop_length

    # Matriz donde cada fila será una trama.
    frames = np.zeros((num_frames, frame_length), dtype=audio.dtype)

    # Extraemos cada bloque.
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i, :] = audio[start:end]

    return frames


def crear_ventana_hamming(N=320):
    """
    Genera una ventana de Hamming de longitud N.
    """
    n = np.arange(N)
    w = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))
    return w.astype(np.float32)


def aplicar_ventana_hamming(frames):
    """
    Multiplica cada trama por una ventana de Hamming.
    """
    N = frames.shape[1]
    ventana = crear_ventana_hamming(N)

    # Multiplicación punto a punto entre cada trama y la ventana.
    frames_windowed = frames * ventana

    return frames_windowed, ventana


def calcular_energia_por_trama(frames):
    """
    Calcula la energía promedio de cada trama.
    """
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def calcular_zcr_por_trama(frames):
    """
    Calcula la tasa de cruces por cero (ZCR) de cada trama.
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
    Detecta el inicio y el final de la voz dentro de un archivo.

    Estrategia usada:
    - Se calcula energía por trama.
    - Se calcula ZCR por trama.
    - Se construyen umbrales relativos.
    - Se marca como voz una trama que supera ambos umbrales.
    """
    energia = calcular_energia_por_trama(frames)
    zcr = calcular_zcr_por_trama(frames)

    # Umbrales relativos respecto al valor máximo del archivo.
    energy_threshold = energy_factor * np.max(energia) if len(energia) > 0 else 0.0
    zcr_threshold = zcr_factor * np.max(zcr) if len(zcr) > 0 else 0.0

    # Se considera voz si una trama supera ambos umbrales.
    voice_flags = (zcr > zcr_threshold) & (energia > energy_threshold)

    # Buscamos tramas marcadas como voz.
    first_voice_frame = np.where(voice_flags)[0]

    # Si no se detectó ninguna, devolvemos None.
    if len(first_voice_frame) == 0:
        return None, None, energia, zcr, voice_flags, energy_threshold, zcr_threshold

    # Primera y última trama con voz.
    inicio_frame = first_voice_frame[0]
    fin_frame = first_voice_frame[-1]

    # Convertimos índices de trama a índices de muestra.
    start_sample = inicio_frame * hop_length
    end_sample = fin_frame * hop_length + frame_length

    return start_sample, end_sample, energia, zcr, voice_flags, energy_threshold, zcr_threshold


def recortar_audio(audio, start_sample, end_sample):
    """
    Recorta la señal entre el inicio y el final detectados.
    """
    if start_sample is None or end_sample is None:
        return None

    end_sample = min(end_sample, len(audio))
    return audio[start_sample:end_sample]


# =========================================================
# AUTOCORRELACIÓN Y LPC
# =========================================================

def autocorrelacion_corta(frame, order=12):
    """
    Calcula la autocorrelación corta de una trama.
    Después de aplicar la ventana, se obtiene la función de autocorrelación
    y a partir de ella se calculan los coeficientes LPC.
    """
    r = np.zeros(order + 1, dtype=np.float64)

    for k in range(order + 1):
        r[k] = np.sum(frame[:len(frame) - k] * frame[k:])

    return r


def levinson_durbin(r, order=12):
    """
    Implementa el algoritmo recursivo de Levinson-Durbin.
    """
    # Vector de coeficientes LPC.
    # a[0] se fija en 1 por convención.
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    # Error inicial igual a r[0].
    e = r[0]

    # Si la energía es casi cero, devolvemos el vector trivial.
    if e <= 1e-12:
        return a, e

    # Construcción recursiva del predictor.
    for i in range(1, order + 1):
        suma = 0.0

        # Parte acumulada de la recursión.
        for j in range(1, i):
            suma += a[j] * r[i - j]

        # Coeficiente de reflexión.
        k = (r[i] - suma) / e

        # Actualización temporal de coeficientes.
        a_new = a.copy()
        a_new[i] = k

        for j in range(1, i):
            a_new[j] = a[j] - k * a[i - j]

        a = a_new

        # Actualización del error.
        e = (1.0 - k * k) * e

        # Evitamos que el error se vuelva numéricamente inestable.
        if e <= 1e-12:
            e = 1e-12
            break

    return a, e


def extraer_lpc_por_trama(frames_hamming, order=12):
    """
    Para cada trama:
    1. calcula la autocorrelación,
    2. obtiene los coeficientes LPC,
    3. guarda también el error de predicción.
    """
    num_frames = frames_hamming.shape[0]

    lpc_vectors = np.zeros((num_frames, order + 1), dtype=np.float64)
    r_vectors = np.zeros((num_frames, order + 1), dtype=np.float64)
    errors = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        frame = frames_hamming[i]

        # Autocorrelación de la trama.
        r = autocorrelacion_corta(frame, order=order)

        # LPC usando Levinson-Durbin.
        a, e = levinson_durbin(r, order=order)

        r_vectors[i, :] = r
        lpc_vectors[i, :] = a
        errors[i] = e

    return lpc_vectors, r_vectors, errors


# =========================================================
# CONVERSIÓN ENTRE LPC Y LSF
# =========================================================

def _unique_angles_sorted(angles, tol=1e-4):
    """
    Ordena ángulos y elimina duplicados casi iguales.

    Esto ayuda a limpiar el resultado numérico durante la conversión a LSF.
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

    Aunque la comparación final se hace con Itakura-Saito,
    el clustering se realiza sobre LSF, porque son más estables
    y convenientes para el agrupamiento.
    """
    a = np.asarray(a, dtype=np.float64)

    if a[0] == 0:
        raise ValueError("El primer coeficiente LPC no puede ser cero.")

    # Normalización para asegurar a[0] = 1.
    a = a / a[0]

    # Construcción del polinomio AR.
    ar = np.concatenate(([1.0], -a[1:]))
    p = len(ar) - 1

    # Construcción de polinomios simétricos y antisimétricos.
    ar_padded = np.concatenate((ar, [0.0]))
    ar_rev = ar_padded[::-1]

    P = ar_padded + ar_rev
    Q = ar_padded - ar_rev

    # División por factores fijos, como se hace en la conversión estándar.
    P_red, _ = np.polydiv(P, np.array([1.0, 1.0]))
    Q_red, _ = np.polydiv(Q, np.array([1.0, -1.0]))

    P_red = np.real_if_close(P_red, tol=1000).astype(np.float64)
    Q_red = np.real_if_close(Q_red, tol=1000).astype(np.float64)

    # Raíces de ambos polinomios.
    roots_P = np.roots(P_red)
    roots_Q = np.roots(Q_red)

    # Extraemos sus ángulos.
    ang_P = np.abs(np.angle(roots_P))
    ang_Q = np.abs(np.angle(roots_Q))

    ang = np.concatenate((ang_P, ang_Q))

    # Conservamos solo los ángulos válidos dentro de (0, pi).
    eps = 1e-6
    ang = ang[(ang > eps) & (ang < np.pi - eps)]

    # Quitamos duplicados numéricos y ordenamos.
    ang = _unique_angles_sorted(ang, tol=1e-4)

    # Validación del número esperado de LSF.
    if len(ang) < p:
        raise ValueError(
            f"No se pudieron obtener suficientes LSF. "
            f"Esperadas: {p}, obtenidas: {len(ang)}"
        )

    if len(ang) > p:
        ang = ang[:p]

    return ang


def _convolve_all(polys):
    """
    Convoluciona una lista de polinomios.
    """
    out = np.array([1.0], dtype=np.float64)

    for p in polys:
        out = np.convolve(out, p)

    return out


def _lsf_to_PQ(lsf):
    """
    Construye los polinomios P y Q a partir de un vector LSF.
    """
    lsf = np.asarray(lsf, dtype=np.float64)
    p = len(lsf)

    if p % 2 != 0:
        raise ValueError("Esta implementación espera orden LPC par.")

    # Separación de frecuencias impares y pares.
    w_odd = lsf[0::2]
    w_even = lsf[1::2]

    P_factors = [
        np.array([1.0, -2.0 * np.cos(w), 1.0], dtype=np.float64)
        for w in w_odd
    ]
    Q_factors = [
        np.array([1.0, -2.0 * np.cos(w), 1.0], dtype=np.float64)
        for w in w_even
    ]

    P = _convolve_all(P_factors)
    Q = _convolve_all(Q_factors)

    P = np.convolve(P, np.array([1.0, 1.0], dtype=np.float64))
    Q = np.convolve(Q, np.array([1.0, -1.0], dtype=np.float64))

    return P, Q


def lsf_a_lpc(lsf):
    """
    Convierte un vector LSF a LPC.

    Esto se necesita porque el agrupamiento se hace en LSF,
    pero la distancia Itakura-Saito se evalúa usando LPC.
    """
    lsf = np.asarray(lsf, dtype=np.float64)
    lsf = np.sort(lsf)

    # Validamos que estén dentro del intervalo correcto.
    if np.any(lsf <= 0) or np.any(lsf >= np.pi):
        raise ValueError("Los LSF deben estar estrictamente dentro de (0, pi).")

    # Validamos que estén estrictamente ordenados.
    if np.any(np.diff(lsf) <= 0):
        raise ValueError("Los LSF deben estar estrictamente ordenados.")

    p = len(lsf)

    if p % 2 != 0:
        raise ValueError("Esta implementación espera orden LPC par.")

    P, Q = _lsf_to_PQ(lsf)

    # Reconstrucción del polinomio A(z).
    A = 0.5 * (P + Q)
    A = A[:-1]
    A = np.real_if_close(A).astype(np.float64)

    if abs(A[0]) < 1e-12:
        raise ValueError("Conversión LSF->LPC inválida: coeficiente líder cero.")

    A = A / A[0]
    return A


def proyectar_lsf_valido(lsf, margen=1e-3):
    """
    Ajusta un vector LSF para que sea válido:
    - dentro de (0, pi),
    - ordenado,
    - sin elementos demasiado pegados.

    Esto ayuda a mantener estabilidad en el entrenamiento del codebook.
    """
    lsf = np.asarray(lsf, dtype=np.float64).copy()
    lsf = np.clip(lsf, margen, np.pi - margen)
    lsf.sort()

    for i in range(1, len(lsf)):
        if lsf[i] <= lsf[i - 1] + margen:
            lsf[i] = lsf[i - 1] + margen

    if lsf[-1] >= np.pi - margen:
        lsf[-1] = np.pi - margen

        for i in range(len(lsf) - 2, -1, -1):
            if lsf[i] >= lsf[i + 1] - margen:
                lsf[i] = lsf[i + 1] - margen

    return lsf


# =========================================================
# PIPELINE COMPLETO PARA UN SOLO AUDIO
# =========================================================

def procesar_audio_a_caracteristicas(
    ruta_audio,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    orden_lpc=12,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65
):
    """
    Procesa un archivo completo desde audio crudo hasta vectores útiles
    para el entrenamiento del sistema.

    Flujo:
    1. leer audio,
    2. aplicar preénfasis,
    3. dividir en tramas,
    4. aplicar Hamming,
    5. detectar voz,
    6. recortar la palabra,
    7. volver a tramificar la señal recortada,
    8. obtener LPC,
    9. convertir a LSF.
    """
    # Cargamos el archivo.
    fs, audio = cargar_audio(ruta_audio)

    # Aplicamos preénfasis.
    audio_pre = aplicar_preenfasis(audio, alpha=alpha)

    # Primera división en tramas para detectar voz.
    frames = dividir_en_tramas(audio_pre, frame_length=frame_length, hop_length=hop_length)

    # Aplicamos Hamming a esas tramas.
    frames_hamming, _ = aplicar_ventana_hamming(frames)

    # Detectamos inicio y fin de la palabra.
    resultado = detectar_inicio_fin(
        frames_hamming,
        hop_length=hop_length,
        frame_length=frame_length,
        energy_factor=energy_factor,
        zcr_factor=zcr_factor
    )

    start_sample, end_sample, energia, zcr, voice_flags, energy_threshold, zcr_threshold = resultado

    # Si no se detectó voz, se detiene.
    if start_sample is None or end_sample is None:
        raise ValueError(f"No se detectó voz en el archivo: {ruta_audio}")

    # Agregamos un margen adicional al inicio y final.
    margen = int((margen_ms / 1000.0) * fs)
    start_sample = max(0, start_sample - margen)
    end_sample = min(len(audio_pre), end_sample + margen)

    # Recortamos la señal.
    audio_recortado = recortar_audio(audio_pre, start_sample, end_sample)

    # Volvemos a tramificar ahora solo la región útil de voz.
    frames_rec = dividir_en_tramas(audio_recortado, frame_length=frame_length, hop_length=hop_length)

    # Aplicamos nuevamente Hamming sobre la señal ya recortada.
    frames_rec_hamming, _ = aplicar_ventana_hamming(frames_rec)

    # Extraemos LPC, autocorrelación y error.
    lpc, r, errors = extraer_lpc_por_trama(frames_rec_hamming, order=orden_lpc)

    # Convertimos cada vector LPC a LSF.
    lsf = []
    indices_validos = []

    for i, a in enumerate(lpc):
        try:
            lsf_i = lpc_a_lsf(a)
            lsf.append(lsf_i)
            indices_validos.append(i)
        except Exception:
            # Si una trama falla en la conversión, se ignora.
            continue

    if len(lsf) == 0:
        raise ValueError(f"No se pudieron obtener vectores LSF en: {ruta_audio}")

    lsf = np.array(lsf, dtype=np.float64)
    indices_validos = np.array(indices_validos, dtype=int)

    # Conservamos solo las tramas válidas.
    lpc = lpc[indices_validos]
    r = r[indices_validos]
    errors = errors[indices_validos]

    return {
        "ruta": ruta_audio,
        "fs": fs,
        "lpc": lpc,
        "r": r,
        "errors": errors,
        "lsf": lsf,
        "num_frames": len(lsf)
    }


# =========================================================
# DISTANCIA ITAKURA-SAITO
# =========================================================

def ra_desde_lpc(a):
    """
    Calcula la autocorrelación corta del vector LPC.
    """
    a = np.asarray(a, dtype=np.float64)
    P = len(a) - 1
    ra = np.zeros(P + 1, dtype=np.float64)

    for i in range(P + 1):
        ra[i] = np.sum(a[:P + 1 - i] * a[i:])

    return ra


def distancia_is_desde_r_y_lpc(r, a, sigma2=1.0, piso=1e-12):
    """
    Calcula la distancia Itakura-Saito entre:
    - la autocorrelación de una trama de prueba,
    - y un modelo LPC de referencia.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    ra = ra_desde_lpc(a)
    P = min(len(r), len(ra)) - 1

    d = (r[0] * ra[0] + 2.0 * np.sum(r[1:P+1] * ra[1:P+1])) / max(sigma2, piso)
    return float(d)


def distancia_total_audio_vs_codebook(r_vectors, centroides_lpc, sigma2_mode="frame"):
    """
    Compara un audio completo contra un codebook.

    Qué hace:
    - para cada trama del audio de prueba,
      calcula la distancia contra todos los centroides del codebook;
    - toma la menor distancia de esa trama;
    - acumula esas mínimas para obtener una distancia total;
    - además calcula la distancia promedio por trama.
    """
    if len(r_vectors) == 0:
        raise ValueError("r_vectors está vacío.")

    dist_total = 0.0

    for i in range(r_vectors.shape[0]):
        r = r_vectors[i]
        sigma2 = max(r[0], 1e-12) if sigma2_mode == "frame" else 1.0

        dists = [
            distancia_is_desde_r_y_lpc(r, centroide, sigma2=sigma2)
            for centroide in centroides_lpc
        ]

        # Se toma la distancia mínima de la trama al codebook.
        dist_total += np.min(dists)

    dist_prom = dist_total / r_vectors.shape[0]
    return dist_total, dist_prom


# =========================================================
# CARGA DE MODELOS ENTRENADOS
# =========================================================

def cargar_modelos_desde_npz(carpeta_modelos, palabras, codebook_size):
    """
    Carga los modelos guardados previamente en archivos NPZ.

    Cada modelo contiene:
    - centroides en LPC,
    - centroides en LSF,
    - y metadatos de entrenamiento.
    """
    modelos = {}

    for palabra in palabras:
        ruta_modelo = os.path.join(
            carpeta_modelos,
            f"{palabra}_codebook_{codebook_size}_is.npz"
        )

        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f"No existe el modelo: {ruta_modelo}")

        data = np.load(ruta_modelo, allow_pickle=True)

        centroides_lpc = data["centroides_lpc"]
        centroides_lsf = data["centroides_lsf"]

        modelos[palabra] = {
            "centroides_lpc": centroides_lpc,
            "centroides_lsf": centroides_lsf,
            "ruta_modelo": ruta_modelo
        }

        print(f"[MODELO] {palabra} cargado desde {ruta_modelo}")

    return modelos


# =========================================================
# CLASIFICACIÓN DE UN SOLO AUDIO
# =========================================================

def clasificar_audio(ruta_audio, modelos, usar_promedio=True):
    """
    Clasifica un archivo de prueba comparándolo contra todos los modelos.

    Qué hace:
    1. procesa el audio de prueba,
    2. obtiene sus vectores de autocorrelación,
    3. calcula la distancia del audio contra cada codebook,
    4. elige la palabra cuyo score sea mínimo.

    Nota:
    - Si usar_promedio=True, se usa distancia promedio por trama.
    - Si usar_promedio=False, se usa distancia total acumulada.
    """
    datos = procesar_audio_a_caracteristicas(
        ruta_audio=ruta_audio,
        alpha=ALPHA_PRENFASIS,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        orden_lpc=ORDEN_LPC,
        energy_factor=ENERGY_FACTOR,
        zcr_factor=ZCR_FACTOR,
        margen_ms=MARGEN_MS
    )

    r_vectors = datos["r"]
    resultados = {}

    for palabra_modelo, modelo in modelos.items():
        dist_total, dist_prom = distancia_total_audio_vs_codebook(
            r_vectors,
            modelo["centroides_lpc"],
            sigma2_mode="frame"
        )

        # Score final para decidir.
        score = dist_prom if usar_promedio else dist_total

        resultados[palabra_modelo] = {
            "dist_total": dist_total,
            "dist_prom": dist_prom,
            "score": score
        }

    # Se elige la palabra con el menor score.
    prediccion = min(resultados, key=lambda k: resultados[k]["score"])
    return prediccion, resultados, datos


# =========================================================
# MATRIZ DE CONFUSIÓN Y REPORTES
# =========================================================

def crear_matriz_confusion(palabras):
    """
    Crea una matriz de confusión vacía.

    Filas:
    palabra real

    Columnas:
    palabra predicha
    """
    n = len(palabras)
    return np.zeros((n, n), dtype=int)


def guardar_matriz_confusion_csv(matriz, palabras, ruta_csv):
    """
    Guarda la matriz de confusión en un archivo CSV.
    """
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real\\pred"] + palabras)

        for i, palabra_real in enumerate(palabras):
            writer.writerow([palabra_real] + list(matriz[i]))


def guardar_reporte_csv(reporte, ruta_csv):
    """
    Guarda un reporte detallado de todos los audios evaluados.

    El reporte incluye:
    - archivo,
    - clase real,
    - clase predicha,
    - si fue correcta o no,
    - número de tramas,
    - distancias contra cada modelo.
    """
    if len(reporte) == 0:
        return

    columnas_base = [
        "archivo",
        "real",
        "predicha",
        "correcta",
        "num_frames"
    ]

    palabras_modelo = sorted(
        [k.replace("dist_total_", "") for k in reporte[0].keys() if k.startswith("dist_total_")]
    )

    columnas_dinamicas = []
    for p in palabras_modelo:
        columnas_dinamicas.extend([
            f"dist_total_{p}",
            f"dist_prom_{p}"
        ])

    columnas = columnas_base + columnas_dinamicas

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in reporte:
            writer.writerow(fila)


# =========================================================
# EVALUACIÓN PARA UN SOLO TAMAÑO DE CODEBOOK
# =========================================================

def probar_tamano_codebook(codebook_size):
    """
    Evalúa todos los audios de prueba para un tamaño específico de codebook.

    Qué hace:
    1. carga los modelos de ese tamaño,
    2. recorre cada carpeta de test,
    3. clasifica cada audio,
    4. actualiza la matriz de confusión,
    5. calcula el accuracy,
    6. guarda reportes.
    """
    carpeta_modelos = f"modelos_codebook_{codebook_size}_is"
    archivo_matriz = f"matriz_confusion_codebook_{codebook_size}_is.csv"
    archivo_reporte = f"reporte_test_codebook_{codebook_size}_is.csv"

    print("\n====================================================")
    print("TEST DE RECONOCIMIENTO CON CODEBOOKS LBG-IS")
    print("====================================================")
    print(f"Orden LPC      : {ORDEN_LPC}")
    print(f"Codevectors    : {codebook_size}")
    print(f"Palabras       : {PALABRAS}")
    print(f"Carpeta modelos: {carpeta_modelos}")
    print("====================================================")

    modelos = cargar_modelos_desde_npz(
        carpeta_modelos=carpeta_modelos,
        palabras=PALABRAS,
        codebook_size=codebook_size
    )

    matriz = crear_matriz_confusion(PALABRAS)
    reporte = []

    total = 0
    correctos = 0

    # Recorremos cada palabra real.
    for i_real, palabra_real in enumerate(PALABRAS):
        rutas_test = sorted(glob.glob(os.path.join(palabra_real, "test", "*.wav")))

        print(f"\n----------------------------------------------------")
        print(f"Evaluando palabra real: {palabra_real}")
        print(f"Archivos test encontrados: {len(rutas_test)}")
        print("----------------------------------------------------")

        for ruta_audio in rutas_test:
            try:
                prediccion, resultados, datos = clasificar_audio(
                    ruta_audio,
                    modelos,
                    usar_promedio=True
                )

                # Índice de la palabra predicha.
                j_pred = PALABRAS.index(prediccion)

                # Se suma en la matriz de confusión.
                matriz[i_real, j_pred] += 1

                es_correcta = (prediccion == palabra_real)
                if es_correcta:
                    correctos += 1
                total += 1

                # Se arma una fila detallada para el reporte.
                fila = {
                    "archivo": ruta_audio,
                    "real": palabra_real,
                    "predicha": prediccion,
                    "correcta": int(es_correcta),
                    "num_frames": int(datos["num_frames"])
                }

                for palabra_modelo in PALABRAS:
                    fila[f"dist_total_{palabra_modelo}"] = resultados[palabra_modelo]["dist_total"]
                    fila[f"dist_prom_{palabra_modelo}"] = resultados[palabra_modelo]["dist_prom"]

                reporte.append(fila)

                print(f"[TEST] {ruta_audio}")
                print(f"       real={palabra_real} | pred={prediccion} | frames={datos['num_frames']}")
                for palabra_modelo in PALABRAS:
                    print(
                        f"       {palabra_modelo:>6s} -> "
                        f"dist_total={resultados[palabra_modelo]['dist_total']:.6f} | "
                        f"dist_prom={resultados[palabra_modelo]['dist_prom']:.6f}"
                    )

            except Exception as e:
                print(f"[ERROR] {ruta_audio} -> {e}")

    accuracy = (correctos / total) * 100.0 if total > 0 else 0.0

    # Impresión de la matriz en consola.
    print("\n====================================================")
    print(f"MATRIZ DE CONFUSION PARA K={codebook_size}")
    print("====================================================")
    print("Filas = palabra real, Columnas = predicción\n")
    print("          " + "  ".join([f"{p:>6s}" for p in PALABRAS]))
    for i, palabra_real in enumerate(PALABRAS):
        fila = "  ".join([f"{matriz[i, j]:6d}" for j in range(len(PALABRAS))])
        print(f"{palabra_real:>6s}  {fila}")

    print("\n====================================================")
    print(f"Accuracy total K={codebook_size}: {accuracy:.2f}%  ({correctos}/{total})")
    print("====================================================")

    guardar_matriz_confusion_csv(matriz, PALABRAS, archivo_matriz)
    guardar_reporte_csv(reporte, archivo_reporte)

    print(f"[GUARDADO] {archivo_matriz}")
    print(f"[GUARDADO] {archivo_reporte}")

    return {
        "codebook_size": codebook_size,
        "accuracy": accuracy,
        "correctos": correctos,
        "total": total,
        "matriz": matriz
    }


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def main():
    """
    Ejecuta la evaluación completa para todos los tamaños de codebook.

    Qué hace:
    - prueba K=16,
    - prueba K=32,
    - prueba K=64,
    - imprime un resumen final comparando accuracies.
    """
    print("====================================================")
    print("TEST MULTIPLE DE RECONOCIMIENTO CON CODEBOOKS LBG-IS")
    print("====================================================")
    print(f"Orden LPC         : {ORDEN_LPC}")
    print(f"Tamaños codebook  : {CODEBOOK_SIZES}")
    print(f"Palabras          : {PALABRAS}")
    print("====================================================")

    resumen = []

    for codebook_size in CODEBOOK_SIZES:
        try:
            resultado = probar_tamano_codebook(codebook_size)
            resumen.append(resultado)
        except Exception as e:
            print(f"\n[ERROR] Falló el test para K={codebook_size}: {e}\n")

    print("\n====================================================")
    print("RESUMEN FINAL")
    print("====================================================")
    for r in resumen:
        print(
            f"K={r['codebook_size']:>2d} | "
            f"Accuracy={r['accuracy']:.2f}% | "
            f"Aciertos={r['correctos']}/{r['total']}"
        )
    print("====================================================")


# Punto de entrada del programa.
if __name__ == "__main__":
    main()