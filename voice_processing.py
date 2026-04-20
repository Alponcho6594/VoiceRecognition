import os
import glob
import json
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
    Calcula la autocorrelación corta de los coeficientes LPC.
    """
    a = np.asarray(a, dtype=np.float64)
    P = len(a) - 1
    ra = np.zeros(P + 1, dtype=np.float64)

    for i in range(P + 1):
        ra[i] = np.sum(a[:P + 1 - i] * a[i:])

    return ra


def distancia_is_desde_r_y_lpc(r, a, sigma2=1.0, piso=1e-12):
    """
    Calcula la distancia Itakura-Saito usando:
    - la autocorrelación de una trama,
    - y un vector LPC de referencia.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    ra = ra_desde_lpc(a)
    P = min(len(r), len(ra)) - 1

    d = (r[0] * ra[0] + 2.0 * np.sum(r[1:P+1] * ra[1:P+1])) / max(sigma2, piso)

    return float(d)


def asignar_clusters_is(r_vectors, centroides_lpc, sigma2_mode="frame"):
    """
    Asigna cada vector de entrenamiento al centroide más cercano
    usando distancia Itakura-Saito.
    """
    N = r_vectors.shape[0]
    K = len(centroides_lpc)

    # Matriz de distancias: filas = vectores, columnas = centroides.
    distancias = np.zeros((N, K), dtype=np.float64)

    for i in range(N):
        r = r_vectors[i]

        # Opcionalmente usamos la energía de la trama como sigma^2.
        sigma2 = max(r[0], 1e-12) if sigma2_mode == "frame" else 1.0

        for k in range(K):
            distancias[i, k] = distancia_is_desde_r_y_lpc(
                r,
                centroides_lpc[k],
                sigma2=sigma2
            )

    # Elegimos el centroide con menor distancia.
    asignaciones = np.argmin(distancias, axis=1)
    dmin = distancias[np.arange(N), asignaciones]

    # Distorsión global.
    distortion_total = np.sum(dmin)

    return asignaciones, distancias, distortion_total


# =========================================================
# ENTRENAMIENTO LBG CON DISTANCIA IS
# =========================================================

def recalcular_centroides_lsf(lsf_vectors, asignaciones, K, centroides_prev=None):
    """
    Recalcula centroides como el promedio de los vectores asignados.
    """
    P = lsf_vectors.shape[1]
    nuevos = np.zeros((K, P), dtype=np.float64)

    for k in range(K):
        idx = np.where(asignaciones == k)[0]

        if len(idx) == 0:
            # Si un cluster queda vacío, se conserva el centroide anterior.
            if centroides_prev is None:
                raise ValueError(f"Cluster vacío en k={k} y no hay centroides_prev.")
            nuevos[k] = centroides_prev[k]
        else:
            nuevos[k] = np.mean(lsf_vectors[idx], axis=0)

    return nuevos


def evaluar_codebook_is(r_vectors, centroides_lsf, sigma2_mode="frame"):
    """
    Evalúa un conjunto de centroides:
    1. fuerza LSF válidos,
    2. convierte centroides a LPC,
    3. asigna vectores con distancia IS,
    4. devuelve la distorsión global.
    """
    centroides_lsf_validos = np.array(
        [proyectar_lsf_valido(c) for c in centroides_lsf],
        dtype=np.float64
    )

    centroides_lpc = [lsf_a_lpc(c) for c in centroides_lsf_validos]

    asignaciones, distancias, D = asignar_clusters_is(
        r_vectors,
        centroides_lpc,
        sigma2_mode=sigma2_mode
    )

    return asignaciones, distancias, D, centroides_lpc, centroides_lsf_validos


def actualizar_con_descenso(
    r_vectors,
    centroides_lsf_old,
    centroides_lsf_prop,
    D_old,
    sigma2_mode="frame",
    max_backtracking=12
):
    """
    Intenta actualizar los centroides sin empeorar la distorsión.

    Si el nuevo promedio propuesto empeora la función objetivo,
    se reduce el paso con backtracking.

    Esto hace al entrenamiento más estable.
    """
    eta = 1.0

    for _ in range(max_backtracking):
        centroides_try = centroides_lsf_old + eta * (centroides_lsf_prop - centroides_lsf_old)
        centroides_try = np.array([proyectar_lsf_valido(c) for c in centroides_try])

        asign, dist, D_try, centroides_lpc_try, centroides_lsf_try = evaluar_codebook_is(
            r_vectors,
            centroides_try,
            sigma2_mode=sigma2_mode
        )

        if D_try <= D_old + 1e-10:
            return centroides_lsf_try, asign, dist, D_try, centroides_lpc_try, eta

        eta *= 0.5

    # Si ninguna actualización mejora, se conservan los centroides previos.
    asign, dist, D_same, centroides_lpc_same, centroides_lsf_same = evaluar_codebook_is(
        r_vectors,
        centroides_lsf_old,
        sigma2_mode=sigma2_mode
    )

    return centroides_lsf_same, asign, dist, D_same, centroides_lpc_same, 0.0


def split_centroides_lsf(centroides_lsf, delta=0.0001):
    """
    Divide cada centroide en dos centroides ligeramente perturbados.
    """
    nuevos = []

    for c in centroides_lsf:
        c1 = proyectar_lsf_valido(c - delta)
        c2 = proyectar_lsf_valido(c + delta)
        nuevos.append(c1)
        nuevos.append(c2)

    return np.array(nuevos, dtype=np.float64)


def entrenar_lbg_is_monotono(
    lsf_vectors,
    r_vectors,
    K_objetivo=16,
    tol_rel=1e-4,
    max_iter=50,
    sigma2_mode="frame",
    delta_split=0.0001,
    verbose=True
):
    """
    Entrena un codebook usando LBG con distancia Itakura-Saito.

    Flujo:
    1. se calcula un centroide inicial,
    2. se divide hasta alcanzar K centroides,
    3. se asignan vectores al centroide más cercano,
    4. se recalculan centroides,
    5. se repite hasta converger.
    """
    # Centroide inicial = promedio de todos los vectores LSF.
    c0 = np.mean(lsf_vectors, axis=0)
    c0 = proyectar_lsf_valido(c0)
    centroides_lsf = np.array([c0], dtype=np.float64)

    historial = []

    # Seguimos duplicando centroides hasta llegar al tamaño deseado.
    while len(centroides_lsf) < K_objetivo:
        centroides_lsf = split_centroides_lsf(centroides_lsf, delta=delta_split)

        # Si por duplicación excedemos el tamaño deseado, recortamos.
        if len(centroides_lsf) > K_objetivo:
            centroides_lsf = centroides_lsf[:K_objetivo]

        # Evaluación inicial del codebook actual.
        asignaciones, distancias, D_old, centroides_lpc, centroides_lsf = evaluar_codebook_is(
            r_vectors,
            centroides_lsf,
            sigma2_mode=sigma2_mode
        )

        if verbose:
            print(f"\n[K={len(centroides_lsf):2d}] Dist inicial = {D_old:.6f}")

        # Refinamiento iterativo.
        for it in range(max_iter):
            # Nuevo promedio por cluster.
            centroides_prop = recalcular_centroides_lsf(
                lsf_vectors,
                asignaciones,
                K=len(centroides_lsf),
                centroides_prev=centroides_lsf
            )

            centroides_prop = np.array([proyectar_lsf_valido(c) for c in centroides_prop])

            # Intentamos actualizar sin empeorar la distorsión.
            centroides_new, asign_new, dist_new, D_new, centroides_lpc_new, eta = actualizar_con_descenso(
                r_vectors=r_vectors,
                centroides_lsf_old=centroides_lsf,
                centroides_lsf_prop=centroides_prop,
                D_old=D_old,
                sigma2_mode=sigma2_mode
            )

            # Cambio relativo de la distorsión.
            rel = abs(D_old - D_new) / max(abs(D_old), 1e-12)

            historial.append({
                "K": int(len(centroides_lsf)),
                "iter": int(it),
                "dist": float(D_new),
                "rel": float(rel),
                "eta": float(eta)
            })

            if verbose:
                print(
                    f"  iter={it:02d} | Dist={D_new:.6f} | Rel={rel:.6e} | eta={eta:.4f}"
                )

            # Actualizamos el estado.
            centroides_lsf = centroides_new
            asignaciones = asign_new
            distancias = dist_new
            centroides_lpc = centroides_lpc_new

            # Criterio de paro.
            if rel < tol_rel:
                break

            D_old = D_new

    return {
        "centroides_lsf": centroides_lsf,
        "centroides_lpc": np.array([lsf_a_lpc(c) for c in centroides_lsf]),
        "asignaciones": asignaciones,
        "distancias": distancias,
        "historial": historial
    }


# =========================================================
# CARGA DE DATOS DE ENTRENAMIENTO POR PALABRA
# =========================================================

def cargar_datos_entrenamiento_palabra(
    palabra,
    orden_lpc=12,
    alpha=0.95,
    frame_length=320,
    hop_length=128,
    energy_factor=0.03,
    zcr_factor=0.08,
    margen_ms=65
):
    """
    Carga todos los audios de entrenamiento de una palabra,
    los procesa y junta todos sus vectores.

    Estructura esperada:
        palabra/train/*.wav
    """
    patron = os.path.join(palabra, "train", "*.wav")
    rutas = sorted(glob.glob(patron))

    if len(rutas) == 0:
        raise FileNotFoundError(f"No se encontraron audios de entrenamiento en: {patron}")

    lsf_all = []
    r_all = []
    archivos_ok = []
    archivos_fallidos = []

    print(f"\n====================================================")
    print(f"Palabra: {palabra}")
    print(f"Archivos train encontrados: {len(rutas)}")
    print(f"====================================================")

    for ruta in rutas:
        try:
            datos = procesar_audio_a_caracteristicas(
                ruta_audio=ruta,
                alpha=alpha,
                frame_length=frame_length,
                hop_length=hop_length,
                orden_lpc=orden_lpc,
                energy_factor=energy_factor,
                zcr_factor=zcr_factor,
                margen_ms=margen_ms
            )

            lsf_all.append(datos["lsf"])
            r_all.append(datos["r"])
            archivos_ok.append(ruta)

            print(f"[OK] {ruta} -> {datos['num_frames']} tramas válidas")

        except Exception as e:
            archivos_fallidos.append((ruta, str(e)))
            print(f"[WARN] {ruta} -> {e}")

    if len(lsf_all) == 0:
        raise RuntimeError(f"No hubo archivos válidos para entrenar la palabra: {palabra}")

    # Unimos todos los vectores de todos los audios en una sola matriz.
    lsf_all = np.vstack(lsf_all)
    r_all = np.vstack(r_all)

    print(f"\nResumen palabra '{palabra}':")
    print(f"  Archivos válidos   : {len(archivos_ok)}")
    print(f"  Archivos fallidos  : {len(archivos_fallidos)}")
    print(f"  Total vectores LSF : {lsf_all.shape[0]}")
    print(f"  Dimensión LSF      : {lsf_all.shape[1]}")

    return {
        "palabra": palabra,
        "lsf": lsf_all,
        "r": r_all,
        "archivos_ok": archivos_ok,
        "archivos_fallidos": archivos_fallidos
    }


def guardar_modelo_palabra(palabra, modelo, carpeta_salida, codebook_size):
    """
    Guarda el modelo entrenado de una palabra.

    Se almacenan:
    - centroides en LSF,
    - centroides en LPC,
    - asignaciones,
    - distancias,
    - historial del entrenamiento.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_npz = os.path.join(carpeta_salida, f"{palabra}_codebook_{codebook_size}_is.npz")
    ruta_json = os.path.join(carpeta_salida, f"{palabra}_historial_{codebook_size}_is.json")

    np.savez(
        ruta_npz,
        palabra=palabra,
        orden_lpc=ORDEN_LPC,
        codebook_size=codebook_size,
        centroides_lsf=modelo["centroides_lsf"],
        centroides_lpc=modelo["centroides_lpc"],
        asignaciones=modelo["asignaciones"],
        distancias=modelo["distancias"]
    )

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(modelo["historial"], f, indent=2, ensure_ascii=False)

    print(f"[GUARDADO] {ruta_npz}")
    print(f"[GUARDADO] {ruta_json}")


def entrenar_codebook_por_palabra(
    palabra,
    orden_lpc=12,
    codebook_size=16,
    carpeta_salida="modelos_codebook_16_is"
):
    """
    Entrena el codebook de una sola palabra.

    Pasos:
    1. carga todos los audios de entrenamiento,
    2. extrae sus características,
    3. entrena un modelo LBG-IS,
    4. guarda el resultado.
    """
    datos = cargar_datos_entrenamiento_palabra(
        palabra=palabra,
        orden_lpc=orden_lpc,
        alpha=ALPHA_PRENFASIS,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        energy_factor=ENERGY_FACTOR,
        zcr_factor=ZCR_FACTOR,
        margen_ms=MARGEN_MS
    )

    print(f"\nEntrenando LBG-IS monotónico para '{palabra}' con K={codebook_size} ...")

    modelo = entrenar_lbg_is_monotono(
        lsf_vectors=datos["lsf"],
        r_vectors=datos["r"],
        K_objetivo=codebook_size,
        tol_rel=TOL_REL,
        max_iter=MAX_ITER_LBG,
        sigma2_mode="frame",
        delta_split=DELTA_SPLIT,
        verbose=True
    )

    guardar_modelo_palabra(palabra, modelo, carpeta_salida, codebook_size)

    return modelo


def entrenar_tamano_codebook(codebook_size):
    """
    Entrena un conjunto completo de modelos para un tamaño de codebook dado.

    Por ejemplo:
    - entrena todos los modelos con K=16,
    - luego con K=32,
    - luego con K=64.
    """
    carpeta_modelos = f"modelos_codebook_{codebook_size}_is"
    os.makedirs(carpeta_modelos, exist_ok=True)

    print("\n====================================================")
    print("ENTRENAMIENTO DE CODEBOOKS LBG-IS")
    print("====================================================")
    print(f"Orden LPC      : {ORDEN_LPC}")
    print(f"Codevectors    : {codebook_size}")
    print(f"Palabras       : {PALABRAS}")
    print(f"Salida modelos : {carpeta_modelos}")
    print("====================================================")

    modelos = {}

    for palabra in PALABRAS:
        try:
            modelo = entrenar_codebook_por_palabra(
                palabra=palabra,
                orden_lpc=ORDEN_LPC,
                codebook_size=codebook_size,
                carpeta_salida=carpeta_modelos
            )
            modelos[palabra] = modelo
            print(f"\n[OK] Entrenamiento completado para '{palabra}' con K={codebook_size}\n")

        except Exception as e:
            print(f"\n[ERROR] No se pudo entrenar '{palabra}' con K={codebook_size}: {e}\n")

    print("====================================================")
    print(f"FIN DEL ENTRENAMIENTO PARA K={codebook_size}")
    print("====================================================")

    return modelos


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def main():
    """
    Ejecuta el entrenamiento completo del sistema.

    Entrena codebooks para todos los tamaños definidos en CODEBOOK_SIZES
    y para todas las palabras del conjunto.
    """
    print("====================================================")
    print("ENTRENAMIENTO MULTIPLE DE CODEBOOKS LBG-IS")
    print("====================================================")
    print(f"Orden LPC         : {ORDEN_LPC}")
    print(f"Tamaños codebook  : {CODEBOOK_SIZES}")
    print(f"Palabras          : {PALABRAS}")
    print("====================================================")

    todos_los_modelos = {}

    for codebook_size in CODEBOOK_SIZES:
        modelos_k = entrenar_tamano_codebook(codebook_size)
        todos_los_modelos[codebook_size] = modelos_k

    print("====================================================")
    print("FIN TOTAL DEL ENTRENAMIENTO")
    print("====================================================")


# Punto de entrada del script.
# Esto hace que el entrenamiento se ejecute solo si el archivo
# se corre directamente desde Python.
if __name__ == "__main__":
    main()