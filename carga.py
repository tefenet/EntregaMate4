import pandas as pd
from urllib.request import urlretrieve
from pathlib import Path
from zipfile import ZipFile

def cargar_datos(ruta_archivo):
    """
    Carga el dataset y asigna nombres a sus columnas.
    El archivo .dat está separado por espacios o tabulaciones.
    """
    nombres_columnas = [
        "frequency",
        "angle",
        "chord_length",
        "velocity",
        "displacement",
        "sound_level"
    ]

    datos = pd.read_csv(
        ruta_archivo,
        sep=r"\s+",
        header=None,
        names=nombres_columnas
    )

    return datos

def carga():
    url_dataset = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"

    directorio_actual = Path.cwd()
    ruta_zip = directorio_actual / "airfoil_self_noise.zip"
    ruta_archivo = directorio_actual / "airfoil_self_noise.dat"

    # Descargar y extraer el dataset si todavía no existe
    if not ruta_archivo.exists():
        urlretrieve(url_dataset, ruta_zip)

        with ZipFile(ruta_zip, "r") as archivo_zip:
            archivo_zip.extract("airfoil_self_noise.dat", directorio_actual)

    return cargar_datos(ruta_archivo)