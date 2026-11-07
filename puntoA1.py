import carga
import numpy as np

def calcular_sumatorias(datos):
    """
    Calcula las sumatorias necesarias para construir
    las tres ecuaciones normales del modelo cuadrático.
    """

    # Variables independiente y dependiente
    frecuencia = datos["frequency"]
    nivel_sonoro = datos["sound_level"]

    # Cantidad de observaciones
    cantidad_observaciones = len(datos)

    # Sumatorias de potencias de x
    suma_x = frecuencia.sum()
    suma_x2 = (frecuencia ** 2).sum()
    suma_x3 = (frecuencia ** 3).sum()
    suma_x4 = (frecuencia ** 4).sum()

    # Sumatorias de y y productos entre x e y
    suma_y = nivel_sonoro.sum()
    suma_xy = (frecuencia * nivel_sonoro).sum()
    suma_x2y = ((frecuencia ** 2) * nivel_sonoro).sum()

    return {
        "n": cantidad_observaciones,
        "suma_x": suma_x,
        "suma_x2": suma_x2,
        "suma_x3": suma_x3,
        "suma_x4": suma_x4,
        "suma_y": suma_y,
        "suma_xy": suma_xy,
        "suma_x2y": suma_x2y
    }


def construir_sistema_ecuaciones(sumatorias):
    """
    Construye la matriz de coeficientes y el vector
    de términos independientes de las ecuaciones normales.
    El sistema tiene la forma:
        A * beta = b
    """

    n = sumatorias["n"]
    sx = sumatorias["suma_x"]
    sx2 = sumatorias["suma_x2"]
    sx3 = sumatorias["suma_x3"]
    sx4 = sumatorias["suma_x4"]

    sy = sumatorias["suma_y"]
    sxy = sumatorias["suma_xy"]
    sx2y = sumatorias["suma_x2y"]

    # Matriz de coeficientes del sistema
    matriz_coeficientes = np.array([
        [n,   sx,  sx2],
        [sx,  sx2, sx3],
        [sx2, sx3, sx4]
    ], dtype=float)

    # Vector de términos independientes
    vector_resultados = np.array([
        sy,
        sxy,
        sx2y
    ], dtype=float)
    return matriz_coeficientes, vector_resultados


def calcular_coeficientes(matriz_coeficientes, vector_resultados):
    """
    Resuelve el sistema de ecuaciones normales.
    Devuelve beta_0, beta_1 y beta_2.
    """
    coeficientes = np.linalg.solve(
        matriz_coeficientes,
        vector_resultados
    )

    return coeficientes
def evaluar_derivada_enX(beta_1, beta_2, valorX):
    return beta_1 + 2 * beta_2 * valorX

def main():
    datos = carga.carga()

    # calcular las sumatorias
    sumatorias = calcular_sumatorias(datos)

    print("SUMATORIAS CALCULADAS")
    for nombre, valor in sumatorias.items():
        print(f"{nombre}: {valor}")

    # construir las ecuaciones normales
    matriz_coeficientes, vector_resultados = (
        construir_sistema_ecuaciones(sumatorias)
    )

    print("\nMATRIZ DE COEFICIENTES")
    print(matriz_coeficientes)

    print("\nVECTOR DE RESULTADOS")
    print(vector_resultados)

    # resolver el sistema
    coeficientes = calcular_coeficientes(
        matriz_coeficientes,
        vector_resultados
    )

    beta_0, beta_1, beta_2 = coeficientes

    print("\nCOEFICIENTES DEL MODELO")
    print(f"beta_0 = {beta_0:.12f}")
    print(f"beta_1 = {beta_1:.12e}")
    print(f"beta_2 = {beta_2:.12e}")
    print(f"derivada en 200 = {evaluar_derivada_enX(beta_1,beta_2,200):.12e}")
    print(f"derivada en 20000 = {evaluar_derivada_enX(beta_1, beta_2, 20000):.12e}")


if __name__ == "__main__":
    main()
