
import numpy as np
import pandas as pd
import carga


def calcular_regresion(datos, variables_explicativas):

    Y = datos["sound_level"].to_numpy()
    X_variables = datos[variables_explicativas].to_numpy()

    n = len(Y)
    k = len(variables_explicativas)

    # Matriz de diseño: primera columna unitaria para el intercepto
    X = np.column_stack((np.ones(n), X_variables))

    # Estimación de coeficientes mediante ecuaciones normales
    beta = np.linalg.solve(X.T @ X, X.T @ Y)

    # Valores ajustados y residuos
    Y_estimado = X @ beta
    residuos = Y - Y_estimado

    # Suma de cuadrados residual (SCE)
    SCE = np.sum(residuos ** 2)

    # Suma de cuadrados total (SCT)
    SCT = np.sum((Y - np.mean(Y)) ** 2)

    # Varianza y desviación estándar residual
    varianza_residual = SCE / (n - k - 1)
    desviacion_residual = np.sqrt(varianza_residual)

    # Coeficiente de determinación
    r2 = 1 - SCE / SCT

    # Coeficiente de determinación ajustado
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    return {
        "beta": beta,
        "n": n,
        "k": k,
        "SCE": SCE,
        "SCT": SCT,
        "varianza_residual": varianza_residual,
        "desviacion_residual": desviacion_residual,
        "r2": r2,
        "r2_ajustado": r2_ajustado,
        "Y_estimado": Y_estimado,
        "residuos": residuos
    }


def construir_cuadro_resultados(resultados):

    filas = []

    for resultado in resultados:

        fila = {
            "Modelo": resultado["nombre"],
            "Variables": resultado["k"],
            "SCE": resultado["SCE"],
            "Varianza residual": resultado["varianza_residual"],
            "Desviación residual": resultado["desviacion_residual"],
            "R²": resultado["r2"],
            "R² ajustado": resultado["r2_ajustado"]
        }

        filas.append(fila)

    return pd.DataFrame(filas)


def mostrar_modelo(resultado):

    beta = resultado["beta"]
    variables = resultado["variables"]

    print("\n" + "=" * 65)
    print(resultado["nombre"])
    print("=" * 65)

    print("\nCoeficientes estimados:")
    print(f"Beta 0 (intercepto): {beta[0]:.6f}")

    for i, variable in enumerate(variables):
        print(f"Beta {i + 1} ({variable}): {beta[i + 1]:.6f}")

    print("\nEcuación del modelo:")

    print(f"sound_level = {beta[0]:.6f}", end="")

    for i, variable in enumerate(variables):
        signo = "+" if beta[i + 1] >= 0 else "-"
        print(
            f" {signo} {abs(beta[i + 1]):.6f} * {variable}",
            end=""
        )

    print("\n")

    print(f"SCE: {resultado['SCE']:.6f}")
    print(f"SCT: {resultado['SCT']:.6f}")
    print(f"Varianza residual: {resultado['varianza_residual']:.6f}")
    print(f"Desviación residual: {resultado['desviacion_residual']:.6f}")
    print(f"R²: {resultado['r2']:.6f}")
    print(f"R² ajustado: {resultado['r2_ajustado']:.6f}")


def main():

    datos = carga.carga()

    # Modelo simple de B.2
    variables_simple = ["frequency"]

    resultado_simple = calcular_regresion(
        datos,
        variables_simple
    )

    resultado_simple["nombre"] = "Modelo simple (B.2)"
    resultado_simple["variables"] = variables_simple

    # Modelo múltiple de B.5
    variables_multiple = [
        "frequency",
        "angle",
        "chord_length",
        "displacement"
    ]

    resultado_multiple = calcular_regresion(
        datos,
        variables_multiple
    )

    resultado_multiple["nombre"] = "Modelo múltiple (B.5)"
    resultado_multiple["variables"] = variables_multiple

    # Mostrar resultados individuales
    mostrar_modelo(resultado_simple)
    mostrar_modelo(resultado_multiple)

    # Construir cuadro comparativo
    resultados = [
        resultado_simple,
        resultado_multiple
    ]

    cuadro = construir_cuadro_resultados(resultados)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    print("\nCUADRO COMPARATIVO DE MODELOS")
    print(cuadro.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Comparación de capacidad explicativa
    diferencia_r2 = (
        resultado_multiple["r2"] - resultado_simple["r2"]
    )

    reduccion_varianza = (
        1 - resultado_multiple["varianza_residual"]
        / resultado_simple["varianza_residual"]
    ) * 100

    print("\nCONCLUSIONES CUANTITATIVAS")
    print(f"Diferencia de R²: {diferencia_r2:.6f}")
    print(f"Reducción de varianza residual: {reduccion_varianza:.2f}%")

    print("\nCOMPARACIÓN DE R² AJUSTADO:")
    print(f"Modelo simple:   {resultado_simple['r2_ajustado']:.6f}")
    print(f"Modelo múltiple: {resultado_multiple['r2_ajustado']:.6f}")

    if resultado_multiple["r2_ajustado"] > resultado_simple["r2_ajustado"]:
        print("\nEl modelo múltiple presenta mejor capacidad explicativa.")
    else:
        print("\nEl modelo simple presenta mejor capacidad explicativa.")


if __name__ == "__main__":
    main()
