import numpy as np
import carga


PREDICTORES = [
    "frequency",
    "angle",
    "chord_length",
    "displacement"
]

RESPUESTA = "sound_level"


def descenso_gradiente(X, y, max_iter=2500000, tolerancia=1e-8):
    """
    Estima los coeficientes de una regresión múltiple
    mediante descenso del gradiente.

    Estandariza los predictores durante el cálculo y
    devuelve los coeficientes en las unidades originales.
    """

    n = len(y)

    # Estandarizar los predictores para facilitar la convergencia
    medias = X.mean(axis=0)
    desviaciones = X.std(axis=0, ddof=0)

    if np.any(desviaciones == 0):
        raise ValueError("Hay un predictor constante; no se puede estandarizar.")
    #si usamos la X como viene, los valores alcanzados son muy lejanos a los optimos
    X_estandarizada = (X - medias) / desviaciones

    # Agregar columna de unos para estimar el intercepto
    X_diseno = np.column_stack([
        np.ones(n),
        X_estandarizada
    ])

    # Función de pérdida: suma de los errores al cuadrado
    def perdida(theta):
        residuos = y - X_diseno @ theta
        return np.sum(residuos**2)

    # Gradiente de la función de pérdida
    def gradiente(theta):
        residuos = y - X_diseno @ theta
        return -2 * X_diseno.T @ residuos

    # Tasa de aprendizaje puesta por prueba y error
    alpha = 0.0002

    # Inicializar todos los coeficientes en cero
    theta = np.zeros(X_diseno.shape[1])
    norma_gradiente_inicial = np.linalg.norm(gradiente(theta))

    for iteracion in range(1, max_iter + 1):
        grad = gradiente(theta)

        # Actualizar todos los coeficientes simultáneamente
        theta_nueva = theta - alpha * grad

        # Criterio de parada: el gradiente ya es muy pequeño
        if np.linalg.norm(grad) <= tolerancia * max(norma_gradiente_inicial, 1):
            theta = theta_nueva
            break

        theta = theta_nueva

    # Convertir los coeficientes a las unidades originales
    coeficientes = theta[1:] / desviaciones
    intercepto = theta[0] - np.sum(theta[1:] * medias / desviaciones)

    # Calcular la pérdida final
    perdida_final = perdida(theta)

    return intercepto, coeficientes, iteracion, perdida_final


def main():
    datos = carga.carga()

    # Mantener las filas completas necesarias para este modelo
    columnas_modelo = PREDICTORES + [RESPUESTA]
    datos = datos.dropna(subset=columnas_modelo).copy()

    # preparar X (predictores) e y (respuesta)
    X = datos[PREDICTORES].to_numpy(dtype=float)
    y = datos[RESPUESTA].to_numpy(dtype=float)

    # estimar los coeficientes por descenso del gradiente
    intercepto, coeficientes, iteraciones, perdida_final = (
        descenso_gradiente(X, y)
    )

    print(f"Observaciones utilizadas: {len(datos)}")
    print(f"Iteraciones realizadas: {iteraciones}")
    print(f"Pérdida final (suma de errores al cuadrado): {perdida_final:.8f}")

    print("\nCoeficientes estimados:")
    print(f"beta_0 (intercepto) = {intercepto:.12f}")

    for nombre, valor in zip(PREDICTORES, coeficientes):
        print(f"Coeficiente de {nombre} = {valor:.12f}")

    print("\nEcuación estimada:")
    print(
        f"sound_level_estimado = {intercepto:.12f}"
        f" + ({coeficientes[0]:.12f}) frequency"
        f" + ({coeficientes[1]:.12f}) angle"
        f" + ({coeficientes[2]:.12f}) chord_length"
        f" + ({coeficientes[3]:.12f}) displacement"
    )
    # Solución exacta (calculo matricial directo con ecuaciones normales)
    X_diseno_original = np.column_stack([np.ones(len(X)), X])
    beta_exacto = np.linalg.inv(X_diseno_original.T @ X_diseno_original) @ X_diseno_original.T @ y
    print("\nSolución exacta (Mínimos Cuadrados):")
    print(beta_exacto)
    beta_estimado = np.concatenate(([intercepto], coeficientes))
    diferencias = beta_estimado - beta_exacto

    print("Diferencias:", diferencias)


if __name__ == "__main__":
    main()
