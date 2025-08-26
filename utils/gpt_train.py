import logging
import os
import gc
import glob
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import shutil
import torch


def detect_and_configure_gpus():
    """
    Automatically detect available GPUs and configure multi-GPU training.
    Returns GPU configuration parameters.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Training will use CPU.")
        return {
            'use_cuda': False,
            'num_gpus': 0,
            'gpu_ids': [],
            'distributed': False
        }
    
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Configure for multi-GPU if more than 1 GPU available
    use_distributed = num_gpus > 1
    
    if use_distributed:
        print(f"Configuring distributed training across {num_gpus} GPUs")
        # Set environment variables for distributed training
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    else:
        print(f"Using single GPU training on GPU 0")
    
    return {
        'use_cuda': True,
        'num_gpus': num_gpus,
        'gpu_ids': gpu_ids,
        'distributed': use_distributed
    }


def find_latest_checkpoint(output_path):
    """
    Find the latest checkpoint in the training directory for resumption.
    Returns the path to the latest checkpoint or None if no checkpoints found.
    """
    # Look for existing training directories
    run_base_dir = os.path.join(output_path, "run")
    if not os.path.exists(run_base_dir):
        return None
    
    # Find all training subdirectories
    training_dirs = []
    for item in os.listdir(run_base_dir):
        item_path = os.path.join(run_base_dir, item)
        if os.path.isdir(item_path) and "training" in item:
            training_dirs.append(item_path)
    
    if not training_dirs:
        return None
    
    # Get the most recent training directory
    latest_training_dir = max(training_dirs, key=os.path.getmtime)
    
    # Find checkpoint files in the latest training directory
    checkpoint_pattern = os.path.join(latest_training_dir, "checkpoint_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by step number (extract number from filename)
    def extract_step(filename):
        try:
            return int(os.path.basename(filename).split('_')[1].split('.')[0])
        except:
            return 0
    
    checkpoint_files.sort(key=extract_step, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def cleanup_old_checkpoints(output_path, max_checkpoints=2):
    """
    Keep only the most recent checkpoints to save storage space.
    Keeps the best_model.pth and the most recent checkpoint.
    """
    # Find all training directories
    run_base_dir = os.path.join(output_path, "run")
    if not os.path.exists(run_base_dir):
        return
    
    training_dirs = []
    for item in os.listdir(run_base_dir):
        item_path = os.path.join(run_base_dir, item)
        if os.path.isdir(item_path):
            training_dirs.append(item_path)
    
    # Clean up checkpoints in each training directory
    for checkpoint_dir in training_dirs:
        # Find all checkpoint files (including checkpoint_1000.pth format)
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= max_checkpoints:
            continue
        
        # Sort by step number (extract number from filename)
        def extract_step(filename):
            try:
                return int(os.path.basename(filename).split('_')[1].split('.')[0])
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step, reverse=True)
        
        # Keep only the most recent max_checkpoints files
        files_to_keep = checkpoint_files[:max_checkpoints]
        files_to_remove = checkpoint_files[max_checkpoints:]
        
        # Also keep best_model.pth and best_model_*.pth files
        best_model_patterns = [
            os.path.join(checkpoint_dir, "best_model.pth"),
            os.path.join(checkpoint_dir, "best_model_*.pth")
        ]
        
        for pattern in best_model_patterns:
            best_files = glob.glob(pattern)
            files_to_keep.extend(best_files)
        
        # Remove old checkpoint files
        for file_path in files_to_remove:
            if file_path not in files_to_keep:
                try:
                    os.remove(file_path)
                    print(f"Removed old checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


class CheckpointCleanupTrainer(Trainer):
    """Custom Trainer that automatically cleans up old checkpoints"""
    
    def __init__(self, *args, **kwargs):
        self.output_base_path = kwargs.pop('output_base_path', None)
        super().__init__(*args, **kwargs)
    
    def save_checkpoint(self, *args, **kwargs):
        # Call the original save_checkpoint method
        result = super().save_checkpoint(*args, **kwargs)
        
        # Clean up old checkpoints after saving
        if self.output_base_path:
            cleanup_old_checkpoints(self.output_base_path, max_checkpoints=2)
        
        return result


def train_gpt(custom_model,version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    # Detect and configure GPUs automatically
    gpu_config = detect_and_configure_gpus()
    
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # print(f"XTTS version = {version}")

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")

    # Training Parameters - Adjust for multi-GPU
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = not gpu_config['distributed']  # False for multi-gpu training
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps
    
    # Adjust batch size for multi-GPU training
    if gpu_config['distributed'] and gpu_config['num_gpus'] > 1:
        print(f"Adjusting batch size for {gpu_config['num_gpus']} GPUs")
        # Keep the same effective batch size by dividing by number of GPUs
        BATCH_SIZE = max(1, batch_size // gpu_config['num_gpus'])
        print(f"Per-GPU batch size: {BATCH_SIZE} (total effective: {BATCH_SIZE * gpu_config['num_gpus']})")


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

    # Adjust num_workers for multi-GPU training
    num_workers = 8
    if language == "ja":
        num_workers = 0
    elif gpu_config['distributed'] and gpu_config['num_gpus'] > 1:
        # Increase workers for multi-GPU to keep GPUs fed with data
        num_workers = min(16, 8 * gpu_config['num_gpus'])
        print(f"Adjusted num_workers to {num_workers} for multi-GPU training")
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
        save_n_checkpoints=2,
        save_checkpoints=True,
        # Multi-GPU configuration
        use_cuda=gpu_config['use_cuda'],
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

    # Check for existing checkpoints to resume training
    latest_checkpoint = find_latest_checkpoint(output_path)
    restore_path = None
    
    if latest_checkpoint:
        print(f"🔄 Resuming training from checkpoint: {os.path.basename(latest_checkpoint)}")
        restore_path = latest_checkpoint
        
        # Extract step number from checkpoint filename for progress tracking
        try:
            step_num = int(os.path.basename(latest_checkpoint).split('_')[1].split('.')[0])
            print(f"📊 Resuming from step: {step_num}")
        except:
            print("⚠️ Could not extract step number from checkpoint filename")
    else:
        print("🆕 Starting fresh training - no existing checkpoints found")
    
    # Configure trainer arguments for multi-GPU
    trainer_args = TrainerArgs(
        restore_path=restore_path,  # Use found checkpoint for resumption
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    )
    
    # Add distributed training configuration if multiple GPUs
    if gpu_config['distributed']:
        trainer_args.use_ddp = True
        trainer_args.rank = 0  # Will be set by DDP launcher
        trainer_args.group_id = "group_id"
        print("Configured for Distributed Data Parallel (DDP) training")
    
    # init the trainer and 🚀
    trainer = CheckpointCleanupTrainer(
        trainer_args,
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        output_base_path=output_path,
    )
    trainer.fit()

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
