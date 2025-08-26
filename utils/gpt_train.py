import logging
import os
import gc
import glob
import re
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import shutil


def cleanup_old_checkpoints(output_dir, max_checkpoints=2):
    """
    Limpia checkpoints antiguos, manteniendo solo los max_checkpoints más recientes.
    Incluye best_model.pth, best_model_*.pth y checkpoint_*.pth
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return
            
        # Patrones de archivos de checkpoint
        checkpoint_patterns = [
            'checkpoint_*.pth',
            'best_model_*.pth'
        ]
        
        all_checkpoints = []
        
        # Recopilar todos los archivos de checkpoint con su tiempo de modificación
        for pattern in checkpoint_patterns:
            for checkpoint_file in output_path.glob(pattern):
                mtime = checkpoint_file.stat().st_mtime
                all_checkpoints.append((checkpoint_file, mtime))
        
        # Ordenar por tiempo de modificación (más reciente primero)
        all_checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Mantener best_model.pth siempre
        best_model_path = output_path / 'best_model.pth'
        protected_files = set()
        if best_model_path.exists():
            protected_files.add(best_model_path)
        
        # Contar archivos a mantener (excluyendo best_model.pth si existe)
        files_to_keep = []
        for checkpoint_file, _ in all_checkpoints:
            if checkpoint_file not in protected_files:
                files_to_keep.append(checkpoint_file)
        
        # Eliminar archivos excedentes
        if len(files_to_keep) > max_checkpoints:
            files_to_delete = files_to_keep[max_checkpoints:]
            for file_to_delete in files_to_delete:
                try:
                    file_to_delete.unlink()
                    print(f" > Deleted old checkpoint: {file_to_delete.name}")
                except OSError as e:
                    print(f" > Warning: Could not delete {file_to_delete.name}: {e}")
                    
    except Exception as e:
        print(f" > Warning: Error during checkpoint cleanup: {e}")


def find_latest_checkpoint(output_dir):
    """
    Encuentra el checkpoint más reciente para reanudar el entrenamiento.
    Retorna la ruta del checkpoint y el número de paso si se encuentra.
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return None, 0
            
        # Buscar checkpoints numerados primero
        checkpoint_files = list(output_path.glob('checkpoint_*.pth'))
        
        if checkpoint_files:
            # Extraer números de paso y encontrar el más alto
            latest_step = 0
            latest_checkpoint = None
            
            for checkpoint_file in checkpoint_files:
                match = re.search(r'checkpoint_(\d+)\.pth', checkpoint_file.name)
                if match:
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = checkpoint_file
            
            if latest_checkpoint:
                return str(latest_checkpoint), latest_step
        
        # Si no hay checkpoints numerados, buscar best_model.pth
        best_model = output_path / 'best_model.pth'
        if best_model.exists():
            return str(best_model), 0
            
        return None, 0
        
    except Exception as e:
        print(f" > Warning: Error finding latest checkpoint: {e}")
        return None, 0


def train_gpt(custom_model,version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # print(f"XTTS version = {version}")

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")
    
    # Verificar si existe un entrenamiento previo para reanudar
    resume_checkpoint = None
    resume_step = 0
    
    if os.path.exists(OUT_PATH):
        # Buscar el directorio de entrenamiento más reciente
        training_dirs = [d for d in os.listdir(OUT_PATH) if os.path.isdir(os.path.join(OUT_PATH, d))]
        if training_dirs:
            # Obtener el directorio más reciente
            latest_dir = max(training_dirs, key=lambda d: os.path.getctime(os.path.join(OUT_PATH, d)))
            latest_training_path = os.path.join(OUT_PATH, latest_dir)
            
            # Buscar checkpoint para reanudar
            resume_checkpoint, resume_step = find_latest_checkpoint(latest_training_path)
            
            if resume_checkpoint:
                print(f" > Found previous training session: {latest_training_path}")
                print(f" > Resuming from checkpoint: {resume_checkpoint} (step {resume_step})")
                
                # Limpiar checkpoints antiguos
                print(" > Cleaning up old checkpoints...")
                cleanup_old_checkpoints(latest_training_path, max_checkpoints=1)
            else:
                print(f" > Previous training directory found but no valid checkpoints: {latest_training_path}")

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps


    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "base_models",f"{version}")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
    XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
    XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
    XTTS_SPEAKER_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file
    XTTS_SPEAKER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_SPEAKER_LINK))  # speakers_xtts.pth file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(f" > Downloading XTTS v{version} files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK,XTTS_SPEAKER_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # Transfer this files to ready folder
    READY_MODEL_PATH = os.path.join(output_path,"ready")
    if not os.path.exists(READY_MODEL_PATH):
        os.makedirs(READY_MODEL_PATH)

    NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
    # NEW_XTTS_CHECKPOINT = os.path.join(READY_MODEL_PATH, "model.pth")
    NEW_XTTS_CONFIG_FILE = os.path.join(READY_MODEL_PATH, "config.json")
    NEW_XTTS_SPEAKER_FILE = os.path.join(READY_MODEL_PATH, "speakers_xtts.pth")

    shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
    # shutil.copy(XTTS_CHECKPOINT, os.path.join(READY_MODEL_PATH, "model.pth"))
    shutil.copy(XTTS_CONFIG_FILE, NEW_XTTS_CONFIG_FILE)
    shutil.copy(XTTS_SPEAKER_FILE, NEW_XTTS_SPEAKER_FILE)

# Use from ready folder
    TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file
    # XTTS_CHECKPOINT = NEW_XTTS_CHECKPOINT  # model.pth file
    XTTS_CONFIG_FILE = NEW_XTTS_CONFIG_FILE  # config.json file
    XTTS_SPEAKER_FILE = NEW_XTTS_SPEAKER_FILE  # speakers_xtts.pth file


    if custom_model != "":
        if os.path.exists(custom_model) and custom_model.endswith('.pth'):
            XTTS_CHECKPOINT = custom_model
            print(f" > Loading custom model: {XTTS_CHECKPOINT}")
        else:
            print(" > Error: The specified custom model is not a valid .pth file path.")

    num_workers = 8
    if language == "ja":
        num_workers = 0
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=num_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=resume_checkpoint,  # Use found checkpoint for resuming training
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # Limpiar checkpoints al final del entrenamiento
    print(" > Final checkpoint cleanup...")
    cleanup_old_checkpoints(trainer.output_path, max_checkpoints=1)

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path
    
    # close file handlers and remove them from the logger
    for handler in logging.getLogger('trainer').handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.getLogger('trainer').removeHandler(handler)
    
    # now you should be able to delete the log file
    log_file = os.path.join(trainer.output_path, f"trainer_{trainer.args.rank}_log.txt")
    os.remove(log_file)

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_SPEAKER_FILE,XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref
