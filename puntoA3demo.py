import numpy as np
import matplotlib.pyplot as plt
import carga

def descenso_gradiente(X, y, alpha, max_iter=2000):
    n = len(y)
    # Estandarización (necesaria para que un solo alpha funcione para todas las variables)
    medias = X.mean(axis=0)
    desviaciones = X.std(axis=0, ddof=0)
    X_est = (X - medias) / desviaciones
    X_diseno = np.column_stack([np.ones(n), X_est])

    # Inicialización en ceros
    theta = np.zeros(X_diseno.shape[1])
    historial = []

    for i in range(max_iter):
        residuos = y - X_diseno @ theta
        perdida = np.sum(residuos ** 2)
        historial.append(perdida)

        # Criterio de detención: si la pérdida ya es muy chica (o cambia muy poco)
        if perdida > 1e15:  # Cortar si explota (divergencia)
            print(f"  [!] Divergencia detectada en iteración {i}")
            break

        grad = -2 * X_diseno.T @ residuos
        theta = theta - alpha * grad

    return theta, historial, medias, desviaciones


datos=carga.carga()
X = datos[["frequency", "angle", "chord_length", "displacement"]].to_numpy(dtype=float)
y = datos["sound_level"].to_numpy(dtype=float)

#3 Tasas de aprendizaje
alphas = {
    "Muy pequeña (1e-7)": 1e-7,
    "Adecuada (0.0002)": 2e-4,
    "Divergente (0.00032)": 3.2e-4  # Depende de los datos, puede necesitar ajuste
}

plt.figure(figsize=(10, 6))

for nombre, alpha in alphas.items():
    print(f"Ejecutando con alpha = {alpha}...")
    _, hist, _, _ = descenso_gradiente(X, y, alpha=alpha)
    plt.plot(hist, label=nombre)

plt.yscale('log')  # Escala logarítmica para ver bien los 3 casos
plt.xlabel("Iteración")
plt.ylabel("Pérdida f (Suma de errores al cuadrado)")
plt.title("Evolución de la pérdida según la tasa de aprendizaje")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()