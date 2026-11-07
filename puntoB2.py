import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import carga

def calcular_regresion_lineal(datos, nombre_variable, valor_x):
    """
    Calcula manualmente una regresión lineal simple
    y sus intervalos de confianza y predicción al 95 %.
    """

    x = datos[nombre_variable].to_numpy(dtype=float)
    y = datos["sound_level"].to_numpy(dtype=float)

    cantidad_datos = len(x)
    grados_libertad = cantidad_datos - 2

    # calcular las medias
    media_x = np.mean(x)
    media_y = np.mean(y)

    # calcular las sumatorias centradas
    suma_cuadrados_x = np.sum((x - media_x) ** 2)
    suma_cuadrados_y = np.sum((y - media_y) ** 2)
    suma_productos_xy = np.sum(
        (x - media_x) * (y - media_y)
    )

    # calcular los coeficientes de la recta
    pendiente = suma_productos_xy / suma_cuadrados_x
    ordenada_origen = media_y - pendiente * media_x

    # calcular las predicciones y los residuos
    valores_estimados = ordenada_origen + pendiente * x
    residuos = y - valores_estimados

    suma_cuadrados_residuos = np.sum(residuos ** 2)

    # varianza residual, R² y correlación
    varianza_residual = (
        suma_cuadrados_residuos / grados_libertad
    )
    if nombre_variable=="frequency":
        estadistico_t = pendiente / np.sqrt(varianza_residual / suma_cuadrados_x)
        print(f"Sxx = {suma_cuadrados_x:.6f}")
        print(f"T = {estadistico_t:.6f}")
    coeficiente_determinacion = (
        1 - suma_cuadrados_residuos / suma_cuadrados_y
    )

    coeficiente_correlacion = (
        suma_productos_xy
        / np.sqrt(suma_cuadrados_x * suma_cuadrados_y)
    )

    # valor crítico de Student para el 95 %
    valor_critico = t.ppf(0.975, grados_libertad)

    # intervalos de confianza de los coeficientes
    error_pendiente = np.sqrt(
        varianza_residual / suma_cuadrados_x
    )

    error_ordenada = np.sqrt(
        varianza_residual * (
            1 / cantidad_datos
            + media_x ** 2 / suma_cuadrados_x
        )
    )

    intervalo_pendiente = (
        pendiente - valor_critico * error_pendiente,
        pendiente + valor_critico * error_pendiente
    )

    intervalo_ordenada = (
        ordenada_origen - valor_critico * error_ordenada,
        ordenada_origen + valor_critico * error_ordenada
    )

    # predicción en x*
    prediccion_x = ordenada_origen + pendiente * valor_x

    factor_incertidumbre = (
        1 / cantidad_datos
        + (valor_x - media_x) ** 2 / suma_cuadrados_x
    )

    # Error estándar de la respuesta media
    error_media = np.sqrt(
        varianza_residual * factor_incertidumbre
    )

    # Error estándar de una nueva observación
    error_prediccion = np.sqrt(
        varianza_residual * (1 + factor_incertidumbre)
    )

    intervalo_media = (
        prediccion_x - valor_critico * error_media,
        prediccion_x + valor_critico * error_media
    )

    intervalo_prediccion = (
        prediccion_x - valor_critico * error_prediccion,
        prediccion_x + valor_critico * error_prediccion
    )

    return {
        "variable": nombre_variable,
        "pendiente": pendiente,
        "ordenada": ordenada_origen,
        "varianza_residual": varianza_residual,
        "r2": coeficiente_determinacion,
        "r": coeficiente_correlacion,
        "ic_pendiente": intervalo_pendiente,
        "ic_ordenada": intervalo_ordenada,
        "ic_media": intervalo_media,
        "ip_prediccion": intervalo_prediccion,
        "min_x": np.min(x),
        "max_x": np.max(x)
    }


def formatear_intervalo(intervalo, decimales=2):
    inferior, superior = intervalo
    return (
        f"[{inferior:.{decimales}f}, "
        f"{superior:.{decimales}f}]"
    )


def construir_cuadro_resultados(resultados):
    filas = []

    for resultado in resultados:
        pendiente = resultado["pendiente"]
        ordenada = resultado["ordenada"]

        # La pendiente lleva 6 decimales; la ordenada, 2.
        ecuacion = f"{pendiente:.6f}x {ordenada:+.2f}"

        filas.append({
            "Xi": resultado["variable"],
            "ŷ = β1x + β0": ecuacion,
            "σ̂²": f"{resultado['varianza_residual']:.2f}",
            "R²": f"{resultado['r2']:.4f}",
            "r": f"{resultado['r']:.4f}",
            "IC(β1)": formatear_intervalo(
                resultado["ic_pendiente"], decimales=5
            ),
            "IC(β0)": formatear_intervalo(
                resultado["ic_ordenada"], decimales=2
            ),
            "ICM(Y|x*)": formatear_intervalo(
                resultado["ic_media"], decimales=2
            ),
            "IP(Y|x*)": formatear_intervalo(
                resultado["ip_prediccion"], decimales=2
            )
        })

    return pd.DataFrame(filas)


def graficar_regresion(datos, resultado):
    """Grafica los datos y la recta de regresión ajustada."""

    nombre_variable = resultado["variable"]

    x = datos[nombre_variable]
    y = datos["sound_level"]

    valores_x_recta = np.linspace(x.min(), x.max(), 200)

    valores_y_recta = (
        resultado["ordenada"]
        + resultado["pendiente"] * valores_x_recta
    )

    plt.figure(figsize=(9, 5))

    plt.scatter(x, y, alpha=0.4, label="Datos observados")

    plt.plot(
        valores_x_recta,
        valores_y_recta,
        color="red",
        linewidth=2,
        label="Recta ajustada"
    )

    plt.xlabel(nombre_variable)
    plt.ylabel("Nivel sonoro (dB)")
    plt.title(f"Regresión lineal: {nombre_variable}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def main():
    datos = carga.carga()

    variables_explicativas = [
        "frequency",
        "angle",
        "chord_length",
        "displacement"
    ]

    resultados = []

    for nombre_variable in variables_explicativas:

        # 2000 Hz solo corresponde a frequency.
        # Para las demás variables, se utiliza su media
        # como valor de referencia, provisionalmente.
        if nombre_variable == "frequency":
            valor_x = 2000
        else:
            valor_x = datos[nombre_variable].mean()

        resultado = calcular_regresion_lineal(
            datos,
            nombre_variable,
            valor_x
        )

        resultados.append(resultado)

    # Mostrar el cuadro solicitado
    cuadro = construir_cuadro_resultados(resultados)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    print("\nCUADRO DE REGRESIONES LINEALES")
    print(cuadro.to_string(index=False))

    # Seleccionar la variable con mayor R²
    mejor_resultado = max(
        resultados,
        key=lambda resultado: resultado["r2"]
    )

    print("\nVARIABLE CON MAYOR R²:")
    print(mejor_resultado["variable"])
    print(f"R² = {mejor_resultado['r2']:.6f}")

    # Graficar la mejor regresión
    graficar_regresion(datos, mejor_resultado)


if __name__ == "__main__":
    main()
