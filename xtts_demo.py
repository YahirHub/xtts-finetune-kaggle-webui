import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import requests
# Imports añadidos para las nuevas funcionalidades
import zipfile
import subprocess
from huggingface_hub import snapshot_download, HfApi, HfFolder
import gdown

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=812):
                f.write(chunk)
        print(f"Archivo descargado en {destination}")
        return destination
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")
        return None

# --- Nuevas Funciones para Descarga, Subida y Comandos ---

def download_direct_link(url, destination_folder):
    """ Descarga un archivo desde una URL directa a una carpeta de destino. """
    if not url:
        return "Por favor, introduce una URL."
    try:
        os.makedirs(destination_folder, exist_ok=True)
        filename = url.split('/')[-1].split('?')[0] or "downloaded_file"
        destination_path = os.path.join(destination_folder, filename)
        
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return f"Archivo descargado exitosamente en {destination_path}"
    except Exception as e:
        return f"Error al descargar: {e}"

def download_huggingface_repo(repo_id, destination_folder, hf_token):
    """ Descarga un repositorio de Hugging Face usando un token opcional. """
    if not repo_id:
        return "Por favor, introduce un ID de repositorio de Hugging Face."
    try:
        os.makedirs(destination_folder, exist_ok=True)
        token = hf_token if hf_token else None
        snapshot_download(repo_id=repo_id, local_dir=destination_folder, local_dir_use_symlinks=False, token=token)
        return f"Repositorio '{repo_id}' descargado exitosamente en {destination_folder}"
    except Exception as e:
        return f"Error al descargar el repositorio: {e}"

def upload_to_huggingface(local_folder, repo_id, hf_token, create_repo, is_private):
    """ Sube una carpeta al Hugging Face Hub. """
    if not all([local_folder, repo_id, hf_token]):
        return "Por favor, completa los campos: Directorio local, ID del Repositorio y Token de HF."
    if not os.path.isdir(local_folder):
        return f"Error: El directorio local '{local_folder}' no existe."
    try:
        api = HfApi()
        if create_repo:
            api.create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=is_private,
                exist_ok=True
            )
        
        repo_url = api.upload_folder(
            folder_path=local_folder,
            repo_id=repo_id,
            token=hf_token,
        )
        return f"¡Directorio subido exitosamente a {repo_url}!"
    except Exception as e:
        return f"Error al subir a Hugging Face: {e}\n{traceback.format_exc()}"

def handle_uploads(files, destination_folder):
    """ Maneja la subida de archivos, descomprimiendo los ZIP. """
    if not files:
        return "Por favor, selecciona archivos para subir."
    try:
        os.makedirs(destination_folder, exist_ok=True)
        log = []
        for temp_file in files:
            temp_path = temp_file.name
            filename = os.path.basename(temp_path)
            destination_path = os.path.join(destination_folder, filename)

            if zipfile.is_zipfile(temp_path):
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(destination_folder)
                log.append(f"Archivo ZIP '{filename}' subido y descomprimido en '{destination_folder}'.")
            else:
                shutil.copy(temp_path, destination_path)
                log.append(f"Archivo '{filename}' subido a '{destination_path}'.")
        return "\n".join(log)
    except Exception as e:
        return f"Error al procesar la subida: {e}\n{traceback.format_exc()}"

def execute_shell_command(command):
    """ Ejecuta un comando de shell y devuelve la salida. """
    if not command:
        return "Por favor, introduce un comando."
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        output = "Salida:\n" + result.stdout
        if result.stderr:
            output += "\nErrores (stderr):\n" + result.stderr
        return output
    except subprocess.CalledProcessError as e:
        return f"Error al ejecutar el comando (código de salida {e.returncode}):\n{e.stderr}"
    except Exception as e:
        return f"Se produjo un error inesperado: {e}"

# Limpiar logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

def clear_gpu_cache():
    # Limpiar la caché de la GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_zip_audios(zip_files, temp_folder="temp_audio_extraction"):
    """
    Extrae archivos de audio de archivos ZIP a una carpeta temporal
    """
    if not zip_files:
        return "No se seleccionaron archivos ZIP.", ""
    
    try:
        # Crear carpeta temporal
        temp_path = os.path.join(tempfile.gettempdir(), temp_folder)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
        
        extracted_count = 0
        supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        for zip_file in zip_files:
            if zipfile.is_zipfile(zip_file.name):
                with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        # Solo extraer archivos de audio
                        if any(file_info.filename.lower().endswith(ext) for ext in supported_formats):
                            # Evitar problemas con nombres de archivos
                            safe_filename = os.path.basename(file_info.filename)
                            if safe_filename:  # Evitar carpetas vacías
                                extract_path = os.path.join(temp_path, safe_filename)
                                with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                                    shutil.copyfileobj(source, target)
                                extracted_count += 1
        
        if extracted_count > 0:
            return f"✅ Extraídos {extracted_count} archivos de audio a carpeta temporal", temp_path
        else:
            return "❌ No se encontraron archivos de audio en los ZIP seleccionados", ""
            
    except Exception as e:
        return f"❌ Error al extraer archivos ZIP: {str(e)}", ""

def download_from_google_drive(gdrive_url, temp_folder="temp_gdrive_download"):
    """
    Descarga un archivo ZIP desde Google Drive usando gdown y lo extrae
    """
    if not gdrive_url:
        return "No se proporcionó URL de Google Drive.", ""
    
    try:
        # Crear carpeta temporal para descarga
        temp_path = os.path.join(tempfile.gettempdir(), temp_folder)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
        
        # Nombre del archivo descargado
        downloaded_file = os.path.join(temp_path, "gdrive_download.zip")
        
        # Descargar usando gdown
        try:
            gdown.download(gdrive_url, downloaded_file, quiet=False, fuzzy=True)
        except Exception as e:
            return f"❌ Error al descargar desde Google Drive: {str(e)}", ""
        
        if not os.path.exists(downloaded_file):
            return "❌ No se pudo descargar el archivo desde Google Drive", ""
        
        # Si es un ZIP, extraer los audios
        if zipfile.is_zipfile(downloaded_file):
            extracted_count = 0
            supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            
            with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Solo extraer archivos de audio
                    if any(file_info.filename.lower().endswith(ext) for ext in supported_formats):
                        # Evitar problemas con nombres de archivos
                        safe_filename = os.path.basename(file_info.filename)
                        if safe_filename:  # Evitar carpetas vacías
                            extract_path = os.path.join(temp_path, safe_filename)
                            with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            extracted_count += 1
            
            # Eliminar el archivo ZIP después de extraer
            os.remove(downloaded_file)
            
            if extracted_count > 0:
                return f"✅ Descargado y extraídos {extracted_count} archivos de audio desde Google Drive", temp_path
            else:
                return "❌ No se encontraron archivos de audio en el ZIP de Google Drive", ""
        else:
            return "❌ El archivo descargado no es un ZIP válido", ""
            
    except Exception as e:
        return f"❌ Error al procesar descarga de Google Drive: {str(e)}", ""

XTTS_MODEL = None

def create_zip(folder_path, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
    return zip_path

def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        return create_zip(ready_folder, "modelo_optimizado")
    return None

def get_dataset_zip(out_path):
    dataset_folder = os.path.join(out_path, "dataset")
    if os.path.exists(dataset_folder):
        return create_zip(dataset_folder, "dataset")
    return None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "¡Necesitas ejecutar los pasos anteriores o establecer manualmente las rutas de checkpoint, config y vocabulario de XTTS!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("¡Cargando modelo XTTS!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("¡Modelo Cargado!")
    return "¡Modelo Cargado!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "¡Necesitas ejecutar el paso anterior para cargar el modelo!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "¡Voz generada!", out_path, speaker_audio_file


def load_params_tts(out_path,version):
    out_path = Path(out_path)
    ready_model_path = out_path / "ready" 

    if not ready_model_path.exists():
        return "El directorio 'ready' no se encuentra en la ruta de salida.", "", "", "", "", ""

    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"
    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          # CORRECCIÓN: Devolver 6 valores en caso de error para coincidir con la salida de Gradio.
          return "Parámetros para TTS no encontrados (model.pth o unoptimize_model.pth)", "", "", "", "", ""        

    return "Parámetros para TTS cargados", str(model_path), str(config_path), str(vocab_path), str(speaker_path), str(reference_path)

# NUEVA FUNCIÓN: Cargar parámetros desde una ruta personalizada
def load_params_from_custom_path(custom_path_str):
    if not custom_path_str or not os.path.isdir(custom_path_str):
        return "La ruta especificada no es un directorio válido.", "", "", "", "", ""

    custom_path = Path(custom_path_str)

    # Definir los nombres de los archivos esperados
    files_to_find = {
        "config": "config.json",
        "vocab": "vocab.json",
        "speaker": "speakers_xtts.pth",
        "reference": "reference.wav"
    }
    
    paths = {}
    missing_files = []

    # Buscar archivos de configuración, vocabulario, etc.
    for key, filename in files_to_find.items():
        file_path = custom_path / filename
        if file_path.exists():
            paths[key] = str(file_path)
        else:
            missing_files.append(filename)

    # Buscar el archivo del modelo (checkpoint)
    model_path_opt = custom_path / "model.pth"
    model_path_unopt = custom_path / "unoptimize_model.pth"
    
    if model_path_opt.exists():
        paths["checkpoint"] = str(model_path_opt)
    elif model_path_unopt.exists():
        paths["checkpoint"] = str(model_path_unopt)
    else:
        missing_files.append("model.pth o unoptimize_model.pth")

    # Si faltan archivos, devolver un error
    if missing_files:
        error_message = f"No se encontraron los siguientes archivos: {', '.join(missing_files)}"
        return error_message, "", "", "", "", ""

    # Devolver todas las rutas encontradas
    return (
        "¡Parámetros cargados desde la ubicación especificada!",
        paths.get("checkpoint", ""),
        paths.get("config", ""),
        paths.get("vocab", ""),
        paths.get("speaker", ""),
        paths.get("reference", "")
    )
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Demo de fine-tuning para XTTS\n\n"""
        """
        Ejemplos de ejecución:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 5003
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        help="Nombre del modelo whisper seleccionado por defecto (Opcional) Opciones: ['large-v3','large-v2', 'large', 'medium', 'small']   Valor por defecto: 'large-v3'",
        default="large-v3",
    )
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        help="Ruta a la carpeta con archivos de audio (opcional)",
        default="",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Habilitar compartir la interfaz de Gradio mediante un enlace público.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Puerto para ejecutar la demo de Gradio. Por defecto: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Ruta de salida (donde se guardarán los datos y checkpoints). Por defecto: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Número de épocas para entrenar. Por defecto: 6",
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Tamaño del lote (batch size). Por defecto: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Pasos de acumulación de gradiente. Por defecto: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Tamaño máximo de audio permitido en segundos. Por defecto: 11",
        default=11,
    )

    args = parser.parse_args()

    with gr.Blocks(title=os.environ.get("APP_NAME", "XTTS Fine-Tuning")) as demo:
        # ... (Tus pestañas 1, 2 y 3 permanecen exactamente iguales)
        with gr.Tab("1 - Procesamiento de Datos"):
            out_path = gr.Textbox(
                label="Ruta de Salida (donde se guardarán datos y checkpoints):",
                value=args.out_path,
            )
            upload_file = gr.File(
                file_count="multiple",
                label="Selecciona aquí los archivos de audio que quieres usar para el entrenamiento de XTTS (Formatos soportados: wav, mp3, y flac)",
            )
            
            # Nueva opción para cargar archivos ZIP con audios
            with gr.Row():
                with gr.Column():
                    zip_upload = gr.File(
                        file_count="multiple",
                        label="O sube archivos ZIP con audios (se extraerán automáticamente):",
                        file_types=[".zip"]
                    )
                    extract_btn = gr.Button("Extraer audios de ZIP", variant="secondary")
                with gr.Column():
                    extraction_status = gr.Textbox(
                        label="Estado de extracción:",
                        interactive=False,
                        placeholder="Selecciona archivos ZIP y presiona 'Extraer audios de ZIP'"
                    )
            
            # Nueva opción para descargar desde Google Drive
            with gr.Row():
                with gr.Column():
                    gdrive_url = gr.Textbox(
                        label="O introduce URL de Google Drive (ZIP con audios):",
                        placeholder="https://drive.google.com/file/d/1ABC123.../view?usp=sharing"
                    )
                    gdrive_download_btn = gr.Button("Descargar desde Google Drive", variant="secondary")
                with gr.Column():
                    gdrive_status = gr.Textbox(
                        label="Estado de descarga:",
                        interactive=False,
                        placeholder="Introduce URL de Google Drive y presiona 'Descargar desde Google Drive'"
                    )
            
            audio_folder_path = gr.Textbox(
                label="Ruta a la carpeta con archivos de audio (opcional):",
                value=args.audio_folder_path,
            )

            whisper_model = gr.Dropdown(
                label="Modelo Whisper",
                value=args.whisper_model,
                choices=[
                    "large-v3",
                    "large-v2",
                    "large",
                    "medium",
                    "small"
                ],
            )

            lang = gr.Dropdown(
                label="Idioma del Dataset",
                value="es",
                choices=[
                    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
                    "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja"
                ],
            )
            progress_data = gr.Label(
                label="Progreso:"
            )

            prompt_compute_btn = gr.Button(value="Paso 1 - Crear Dataset")
        
            def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
            
                train_csv = ""
                eval_csv = ""
            
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
            
                if audio_folder_path:
                    audio_files = list(list_audios(audio_folder_path))
                else:
                    audio_files = audio_path
            
                if not audio_files:
                    return "¡No se encontraron archivos de audio! Por favor, sube archivos o especifica una ruta de carpeta.", "", ""
                else:
                    try:
                        # Cargando Whisper
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        asr_model = None
                        if torch.cuda.is_available():
                            try:
                                # Intentar con float16, el más rápido
                                print("Intentando cargar el modelo Whisper con compute_type='float16'...")
                                asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")
                                print("Modelo Whisper cargado exitosamente con float16.")
                            except ValueError as e:
                                if "float16" in str(e):
                                    print(f"ADVERTENCIA: La GPU actual no soporta 'float16' de manera eficiente. Cambiando a 'int8'. Error: {e}")
                                    try:
                                        # Si float16 falla, intentar con int8
                                        print("Intentando cargar el modelo Whisper con compute_type='int8'...")
                                        asr_model = WhisperModel(whisper_model, device=device, compute_type="int8")
                                        print("Modelo Whisper cargado exitosamente con int8.")
                                    except Exception as e_int8:
                                        print(f"ERROR: Falló la carga del modelo Whisper con 'int8' también. Error: {e_int8}")
                                        traceback.print_exc()
                                        error = traceback.format_exc()
                                        return f"Error al cargar el modelo Whisper. Ni float16 ni int8 funcionaron. Revisa la consola.\nResumen del error: {error}", "", ""
                                else:
                                    # Otro ValueError no relacionado con float16
                                    raise e
                        else:
                            # Usar float32 para CPU
                            print("No se detectó GPU. Cargando modelo Whisper en CPU con compute_type='float32'...")
                            asr_model = WhisperModel(whisper_model, device=device, compute_type="float32")
                            print("Modelo Whisper cargado exitosamente en CPU.")

                        if asr_model is None:
                            return "No se pudo cargar el modelo de Whisper. Revisa la consola para ver los errores.", "", ""

                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path, gradio_progress=progress)

                    except Exception:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"¡El procesamiento de datos fue interrumpido por un error! Por favor, revisa la consola para ver el mensaje completo.\nResumen del error: {error}", "", ""
            
                if audio_total_size < 120:
                    message = "¡La duración total de los audios que proporcionaste debe ser de al menos 2 minutos!"
                    print(message)
                    return message, "", ""
            
                print("¡Dataset Procesado!")
                return "¡Dataset Procesado!", train_meta, eval_meta

        with gr.Tab("2 - Fine-tuning (Afinamiento) del Codificador XTTS"):
            load_params_btn = gr.Button(value="Cargar Parámetros desde la carpeta de salida")
            version = gr.Dropdown(
                label="Versión base de XTTS",
                value="v2.0.2",
                choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"],
            )
            train_csv = gr.Textbox(label="CSV de Entrenamiento:")
            eval_csv = gr.Textbox(label="CSV de Evaluación:")
            custom_model = gr.Textbox(
                label="(Opcional) Archivo model.pth personalizado, déjalo en blanco para usar el modelo base.",
                value="",
            )
            num_epochs =  gr.Slider(
                label="Número de épocas:", minimum=1, maximum=100, step=1, value=args.num_epochs
            )
            batch_size = gr.Slider(
                label="Tamaño del lote (Batch size):", minimum=2, maximum=512, step=1, value=args.batch_size
            )
            grad_acumm = gr.Slider(
                label="Pasos de acumulación de gradiente:", minimum=2, maximum=128, step=1, value=args.grad_acumm
            )
            max_audio_length = gr.Slider(
                label="Tamaño máximo de audio permitido (segundos):", minimum=2, maximum=20, step=1, value=args.max_audio_length
            )
            clear_train_data = gr.Dropdown(
                label="Limpiar datos de entrenamiento después de optimizar (eliminará la carpeta seleccionada)",
                value="none",
                choices=["none", "run", "dataset", "all"]
            )
            
            progress_train = gr.Label(label="Progreso:")
            train_btn = gr.Button(value="Paso 2 - Iniciar Entrenamiento")
            optimize_model_btn = gr.Button(value="Paso 2.5 - Optimizar el Modelo")
            
            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                clear_gpu_cache()
          
                if custom_model.startswith("http"):
                    print("Descargando modelo personalizado desde URL...")
                    custom_model = download_file(custom_model, "custom_model.pth")
                    if not custom_model:
                        return "Fallo al descargar el modelo personalizado.", "", "", "", ""
            
                run_dir = Path(output_path) / "run"
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("El idioma preparado para el dataset no coincide. Se cambiará el idioma al especificado en el dataset.")
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "¡Necesitas ejecutar el paso de procesamiento de datos o establecer manualmente las rutas de los CSV!", "", "", "", ""
                try:
                    max_audio_length = int(max_audio_length * 22050)
                    speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"¡El entrenamiento fue interrumpido por un error! Revisa la consola para ver el mensaje completo.\nResumen del error: {error}", "", "", "", ""
            
                ready_dir = Path(output_path) / "ready"
                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
                ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
            
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)
            
                print("¡Entrenamiento del modelo finalizado!")
                return "¡Entrenamiento del modelo finalizado!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

            def optimize_model(out_path, clear_train_data):
                out_path = Path(out_path)
                ready_dir = out_path / "ready"
                run_dir = out_path / "run"
                dataset_dir = out_path / "dataset"
            
                if clear_train_data in {"run", "all"} and run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except PermissionError as e:
                        print(f"Ocurrió un error al eliminar {run_dir}: {e}")
            
                if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
                    try:
                        shutil.rmtree(dataset_dir)
                    except PermissionError as e:
                        print(f"Ocurrió un error al eliminar {dataset_dir}: {e}")
            
                model_path = ready_dir / "unoptimize_model.pth"
                if not model_path.is_file():
                    return "Modelo no optimizado no encontrado en la carpeta 'ready'", ""
            
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                del checkpoint["optimizer"]
                for key in list(checkpoint["model"].keys()):
                    if "dvae" in key:
                        del checkpoint["model"][key]
                os.remove(model_path)

                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)
                clear_gpu_cache()
        
                return f"¡Modelo optimizado y guardado en {ft_xtts_checkpoint}!", ft_xtts_checkpoint

            def load_params(out_path):
                path_output = Path(out_path)
                dataset_path = path_output / "dataset"
                if not dataset_path.exists():
                    return "¡La carpeta de salida no existe!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"
                lang_file_path =  dataset_path / "lang.txt"

                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                clear_gpu_cache()
                print(current_language)
                return "Los datos han sido actualizados", eval_train, eval_csv, current_language

        with gr.Tab("3 - Inferencia"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button(value="Cargar parámetros para TTS desde la carpeta de salida")
                    
                    # NUEVOS COMPONENTES DE LA INTERFAZ
                    gr.Markdown("---")
                    gr.Markdown("O bien, carga desde una ubicación específica:")
                    custom_model_path_input = gr.Textbox(
                        label="Ruta de la carpeta del modelo",
                        placeholder="Ej: /ruta/a/mi/modelo_listo/"
                    )
                    load_from_custom_path_btn = gr.Button(value="Cargar desde ubicación específica")
                    gr.Markdown("---")

                    xtts_checkpoint = gr.Textbox(label="Ruta del checkpoint XTTS:")
                    xtts_config = gr.Textbox(label="Ruta del config XTTS:")
                    xtts_vocab = gr.Textbox(label="Ruta del vocabulario XTTS:")
                    xtts_speaker = gr.Textbox(label="Ruta del speaker XTTS:")
                    progress_load = gr.Label(label="Progreso:")
                    load_btn = gr.Button(value="Paso 3 - Cargar Modelo XTTS Afinado")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(label="Audio de referencia del hablante:")
                    tts_language = gr.Dropdown(
                        label="Idioma",
                        value="es",
                        choices=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","hu","ko","ja"],
                    )
                    tts_text = gr.Textbox(
                        label="Texto de Entrada.",
                        value="Este modelo suena realmente bien y, sobre todo, es razonablemente rápido.",
                    )
                    with gr.Accordion("Configuración Avanzada", open=False) as acr:
                        temperature = gr.Slider(label="temperature", minimum=0, maximum=1, step=0.05, value=0.75)
                        length_penalty  = gr.Slider(label="length_penalty", minimum=-10.0, maximum=10.0, step=0.5, value=1)
                        repetition_penalty = gr.Slider(label="repetition penalty", minimum=1, maximum=10, step=0.5, value=5)
                        top_k = gr.Slider(label="top_k", minimum=1, maximum=100, step=1, value=50)
                        top_p = gr.Slider(label="top_p", minimum=0, maximum=1, step=0.05, value=0.85)
                        sentence_split = gr.Checkbox(label="Habilitar división de texto por frases", value=True)
                        use_config = gr.Checkbox(label="Usar configuración de inferencia del archivo config (ignora los ajustes de arriba)", value=False)
                    
                    tts_btn = gr.Button(value="Paso 4 - Generar Audio (Inferencia)")
                    model_download_btn = gr.Button("Paso 5 - Descargar Modelo Optimizado (ZIP)")
                    dataset_download_btn = gr.Button("Paso 5 - Descargar Dataset (ZIP)")
                
                    model_zip_file = gr.File(label="Descargar Modelo Optimizado", interactive=False)
                    dataset_zip_file = gr.File(label="Descargar Dataset", interactive=False)

                with gr.Column() as col3:
                    progress_gen = gr.Label(label="Progreso:")
                    tts_output_audio = gr.Audio(label="Audio Generado.")
                    reference_audio = gr.Audio(label="Audio de Referencia Usado.")
        
        # --- PESTAÑA DE UTILIDADES MEJORADA ---
        with gr.Tab("Descargas y Utilidades"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Descarga Directa")
                    direct_url = gr.Textbox(label="URL del archivo", placeholder="https://example.com/file.zip")
                    direct_dest = gr.Textbox(label="Carpeta de destino", value="./downloads")
                    direct_btn = gr.Button("Descargar archivo")
                    direct_status = gr.Label(label="Estado:")
                with gr.Column():
                    gr.Markdown("### Subir Archivos Locales")
                    upload_files_utility = gr.File(
                        label="Subir archivos (los .zip se descomprimirán automáticamente)",
                        file_count="multiple"
                    )
                    upload_dest = gr.Textbox(label="Carpeta de destino para subir", value="./uploads")
                    upload_btn = gr.Button("Subir y procesar archivos")
                    upload_status = gr.Label(label="Estado:")
            
            with gr.Blocks():
                gr.Markdown("---")
                gr.Markdown("## Utilidades de Hugging Face")
                hf_token = gr.Textbox(label="Token de Hugging Face", type="password", placeholder="hf_...")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Descargar Repositorio")
                        hf_repo_down = gr.Textbox(label="ID del Repositorio a descargar", placeholder="coqui/XTTS-v2")
                        hf_dest_down = gr.Textbox(label="Carpeta de destino", value="./hf_models")
                        hf_btn_down = gr.Button("Descargar repositorio")
                        hf_status_down = gr.Label(label="Estado:")
                    with gr.Column():
                        gr.Markdown("### Subir Directorio a HF Hub")
                        hf_folder_up = gr.Textbox(label="Directorio local a subir", placeholder="./finetune_models/ready")
                        hf_repo_up = gr.Textbox(label="ID del Repositorio en HF (se creará si no existe)", placeholder="tu-usuario/nombre-del-repo")
                        with gr.Row():
                            hf_create_repo = gr.Checkbox(label="Crear repositorio si no existe", value=True)
                            hf_is_private = gr.Checkbox(label="Privado", value=False)
                        hf_btn_up = gr.Button("Subir a Hugging Face")
                        hf_status_up = gr.Label(label="Estado:")

            with gr.Blocks():
                gr.Markdown("---")
                gr.Markdown("## Ejecutar Comandos de Shell")
                gr.Markdown("⚠️ **Aviso:** La ejecución de comandos de terminal desde una interfaz web es extremadamente peligrosa. Úsalo bajo tu propio riesgo y solo si entiendes completamente lo que estás haciendo.")
                shell_command = gr.Textbox(label="Comando", placeholder="ls -l ./finetune_models")
                shell_run_btn = gr.Button("Ejecutar comando")
                shell_output = gr.Textbox(label="Salida del Comando", lines=10, interactive=False)
                shell_run_btn.click(fn=execute_shell_command, inputs=[shell_command], outputs=[shell_output])

        # --- Conexiones para la nueva pestaña ---
        direct_btn.click(fn=download_direct_link, inputs=[direct_url, direct_dest], outputs=[direct_status])
        upload_btn.click(fn=handle_uploads, inputs=[upload_files_utility, upload_dest], outputs=[upload_status])
        hf_btn_down.click(fn=download_huggingface_repo, inputs=[hf_repo_down, hf_dest_down, hf_token], outputs=[hf_status_down])
        hf_btn_up.click(fn=upload_to_huggingface, inputs=[hf_folder_up, hf_repo_up, hf_token, hf_create_repo, hf_is_private], outputs=[hf_status_up])
        
        # Función para manejar la extracción de ZIP
        def handle_zip_extraction(zip_files):
            status, extracted_path = extract_zip_audios(zip_files)
            return status, extracted_path
        
        extract_btn.click(
            fn=handle_zip_extraction,
            inputs=[zip_upload],
            outputs=[extraction_status, audio_folder_path]
        )
        
        # Función para manejar la descarga de Google Drive
        def handle_gdrive_download(gdrive_url):
            status, extracted_path = download_from_google_drive(gdrive_url)
            return status, extracted_path
        
        gdrive_download_btn.click(
            fn=handle_gdrive_download,
            inputs=[gdrive_url],
            outputs=[gdrive_status, audio_folder_path]
        )

        load_params_btn.click(
            fn=load_params,
            inputs=[out_path],
            outputs=[progress_data, train_csv, eval_csv, lang],
        )

        prompt_compute_btn.click(
            fn=preprocess_dataset,
            inputs=[upload_file, audio_folder_path, lang, whisper_model, out_path, train_csv, eval_csv],
            outputs=[progress_data, train_csv, eval_csv],
        )

        train_btn.click(
            fn=train_model,
            inputs=[custom_model, version, lang, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, out_path, max_audio_length],
            outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
        )

        optimize_model_btn.click(
            fn=optimize_model,
            inputs=[out_path, clear_train_data],
            outputs=[progress_train,xtts_checkpoint],
        )
            
        load_btn.click(
            fn=load_model,
            inputs=[xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker],
            outputs=[progress_load],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[tts_language, tts_text, speaker_reference_audio, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config],
            outputs=[progress_gen, tts_output_audio,reference_audio],
        )

        load_params_tts_btn.click(
            fn=load_params_tts,
            inputs=[out_path, version],
            outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
        )
        
        # NUEVA CONEXIÓN para el botón de carga desde ruta personalizada
        load_from_custom_path_btn.click(
            fn=load_params_from_custom_path,
            inputs=[custom_model_path_input],
            outputs=[progress_load, xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker, speaker_reference_audio]
        )
         
        model_download_btn.click(
            fn=get_model_zip,
            inputs=[out_path],
            outputs=[model_zip_file]
        )
        
        dataset_download_btn.click(
            fn=get_dataset_zip,
            inputs=[out_path],
            outputs=[dataset_zip_file]
        )

    demo.launch(
        share=args.share,
        debug=False,
        server_port=args.port,
    )