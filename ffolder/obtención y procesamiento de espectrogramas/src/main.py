# ================================ Librerías ================================ #
from SpectrogramOfVoice import SpectrogramOfVoice

# ================================== Main =================================== #
# Instancia de la clase
specvo = SpectrogramOfVoice()

# Descargar y descomprimir los datasets deseados
# specvo.descargar_dataset_kaggle( "bryanpark/spanish-single-speaker-speech-dataset", "spanish_speech")

# specvo.descargar_dataset_kaggle("bryanpark/chinese-single-speaker-speech-dataset", "chinese_speech")

# +++++++++++++++++++++++++++++++++ Chino +++++++++++++++++++++++++++++++++ #
# Parámetros de entrada y salida
input_base_dir = "chinese_speech"
output_base_dir = "chinese_spectrogram_32x32"
subcarpetas = ["call_to_arms", "chao_hua_si_she", "zh"]

# Generar espectrogramas de 32x32 píxeles en el directorio especificado
specvo.generar_espectrogramas_en_directorio(
    input_base_dir, output_base_dir, subcarpetas, output_size=32)

# Generar espectrogramas de 64x64 píxeles en el directorio especificado
output_base_dir = "chinese_spectrogram_64x64"
specvo.generar_espectrogramas_en_directorio(
    input_base_dir, output_base_dir, subcarpetas, output_size=64)

# +++++++++++++++++++++++++++++++++ Español +++++++++++++++++++++++++++++++++ #
# Parámetros de entrada y salida
input_base_dir = "spanish_speech"
output_base_dir = "spanish_spectrogram_32x32"
subcarpetas = ["19demarzo", "bailen", "batalla_arapiles"]

# Generar espectrogramas de 32x32 píxeles en el directorio especificado
specvo.generar_espectrogramas_en_directorio(
    input_base_dir, output_base_dir, subcarpetas, output_size=32)

# Generar espectrogramas de 64x64 píxeles en el directorio especificado
output_base_dir = "chinese_spectrogram_64x64"
specvo.generar_espectrogramas_en_directorio(
    input_base_dir, output_base_dir, subcarpetas, output_size=64)
