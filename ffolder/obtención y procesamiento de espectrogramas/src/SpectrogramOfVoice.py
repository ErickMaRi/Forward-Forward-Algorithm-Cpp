# ================================ Librerías ================================ #
import subprocess
import zipfile
import os
import librosa
import numpy as np
from PIL import Image  # Importar PIL para manejar imágenes

# ================================== Clase ================================== #
class SpectrogramOfVoice:
    """
    Clase para gestionar la descarga y extracción de datasets de voz desde
    Kaggle, para luego transformar archivos .wav a imágenes de espectrogramas.
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def __init__(self):
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def descargar_dataset_kaggle(self, dataset_id, output_folder,
                                 no_remove=False):
        """
        Descarga y descomprime un dataset de Kaggle, si no existe la carpeta
        de destino.

        Args:
            dataset_id (str): El identificador del dataset en Kaggle.
            output_folder (str): La carpeta donde se descomprimirá el
                                 contenido del dataset.
            no_remove (bool): Se borra el .zip si es False.

        Raises:
            FileNotFoundError: Si el archivo .zip no se encuentra después de
                               la descarga.
        """
        # Verificar si la carpeta de destino ya existe
        if os.path.exists(output_folder):
            print(f"La carpeta '{output_folder}' ya existe. Se omite la "
                  f"descarga del dataset {dataset_id}.")
            return  # Termina la función si la carpeta ya existe

        print(f"Descargando el dataset {dataset_id}...")

        # Ejecuta el comando de descarga usando Kaggle API
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id])

        # Nombre del archivo zip basado en el ID del dataset
        zip_file = dataset_id.split("/")[-1] + ".zip"

        # Verifica si el archivo descargado existe y procede a descomprimir
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            print(f"Dataset {dataset_id} descomprimido en la carpeta "
                  f"'{output_folder}' correctamente.")

            # Elimina el archivo .zip para liberar espacio
            if no_remove is False:
                os.remove(zip_file)
                print(f"Archivo {zip_file} eliminado para liberar espacio.")
        else:
            raise FileNotFoundError(f"Error: El archivo {zip_file} no se "
                                    "encontró.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def generar_espectrograma(self, archivo_wav, output_imagen,
                              output_size=32):
        """
        Genera un espectrograma en blanco y negro de tamaño especificado a partir de un archivo .wav
        y lo guarda como imagen.

        Args:
            archivo_wav (str): Ruta del archivo de audio .wav.
            output_imagen (str): Ruta donde se guardará la imagen del espectrograma.
            output_size (int): Tamaño en píxeles de la imagen de salida (32 o 64).
        """
        # Crear el directorio de salida si no existe
        output_dir = os.path.dirname(output_imagen)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio '{output_dir}' creado.")

        # Cargar el archivo de audio
        y, sr = librosa.load(archivo_wav, sr=None)

        # Generar el espectrograma usando la STFT
        stft = librosa.stft(y)
        espectrograma_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # Normalizar los valores del espectrograma a rango 0-255
        norm_espectrograma = (espectrograma_db - espectrograma_db.min()) / (espectrograma_db.max() - espectrograma_db.min())
        norm_espectrograma = (norm_espectrograma * 255).astype(np.uint8)

        # Convertir el espectrograma en una imagen PIL
        img = Image.fromarray(norm_espectrograma)
        img = img.convert('L')  # Asegurar que es escala de grises

        # Redimensionar la imagen al tamaño deseado
        img_resized = img.resize((output_size, output_size))

        # Guardar la imagen
        img_resized.save(output_imagen)
        print(f"Espectrograma guardado en '{output_imagen}'.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def generar_espectrogramas_en_directorio(self, input_base_dir,
                                             output_base_dir, subcarpetas,
                                             output_size=32):
        """
        Genera espectrogramas para todos los archivos .wav en las subcarpetas
        especificadas y los guarda en la estructura de salida correspondiente.

        Args:
            input_base_dir (str): Directorio base de entrada que contiene las
                                  subcarpetas con archivos .wav.
            output_base_dir (str): Directorio base de salida donde se guardarán
                                   los espectrogramas generados.
            subcarpetas (list): Lista de nombres de subcarpetas que contienen
                                archivos .wav.
            output_size (int): Tamaño en píxeles de la imagen de salida (32 o 64).
        """
        for subcarpeta in subcarpetas:
            # Define el directorio de entrada donde están los archivos .wav
            input_dir = os.path.join(input_base_dir, subcarpeta)

            # Define el directorio de salida donde se guardarán los espectrogramas
            output_dir = os.path.join(output_base_dir, subcarpeta)

            # Verifica si el directorio de salida existe; si no, lo crea
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Directorio de salida '{output_dir}' creado.")

            # Recorre todos los archivos en el directorio de entrada
            for archivo in os.listdir(input_dir):
                # Procesa solo los archivos que terminan en .wav
                if archivo.endswith(".wav"):
                    # Ruta completa del archivo .wav de entrada
                    archivo_wav = os.path.join(input_dir, archivo)

                    # Define el sufijo del archivo de salida
                    suffix = ".png"

                    # Define la ruta de salida para la imagen del espectrograma
                    output_imagen = os.path.join(
                        output_dir,
                        f"espectrograma_{archivo.replace('.wav', suffix)}")

                    # Genera el espectrograma del archivo .wav y lo guarda como imagen
                    self.generar_espectrograma(
                        archivo_wav, output_imagen, output_size=output_size)
