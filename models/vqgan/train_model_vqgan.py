"""
Optuna-based hyperparameter optimization for VQGAN / MedicalDiffusion models
with PyTorch Lightning and MLflow integration.

This script:
- Loads experiment configuration from YAML
- Prepares dataset(s)
- Defines an Optuna objective function for training models
- Runs Optuna study for hyperparameter search
- Logs results (CSV, MLflow, checkpoints)
"""
import os
import sys
import shutil
import argparse
import joblib
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
import optuna

from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

# === Project-specific imports ===
from dataset.cnn_diffusion.dataset_preprocessing import prepare_dataset
from models.vqgan.vqgan_model import VQGAN
from models.vqgan.LPIPS_callback import LPIPSTopNModels3D
from models.vqgan.MSSSIM_callback import MSSSIMTopNModels3D
from models.vqgan.FID_callback import FIDTopNModels3D
from models.vqgan.multimetric_callback import MultiMetricTopNModels3D
from models.mlflow_wrapper import MLflowWrapper


# Make CUDA allocations expandable to avoid OOM errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# Objective function for Optuna
# =========================================================
def objective(trial, trials_config, train_loader, validation_loader):
    """
    Optuna objective function. Defines how a single trial is run:
    - Suggests hyperparameters
    - Builds model and training callbacks
    - Runs training with PyTorch Lightning
    - Returns validation loss for optimization
    """

    # --- Hyperparameter suggestions ---
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    batch_size_sample = trial.suggest_categorical("batch_size_sample", trials_config["batch_size_sample"])

    config["batch_size_train"] = batch_size_train
    config["batch_size_sample"] = batch_size_sample

    # Handle dataset resampling per trial if configured
    if trials_config.get("n_train_samples") is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples
        config["n_test_samples"] = trials_config.get("n_test_samples")

        train_set, validation_set, test_set = prepare_dataset(
            study, config,
            data_dir=data_dir,
            data_file_name=trials_config["data_file_name"],
            serialize_path=output_dir
        )

        print(f"len(trainset): {len(train_set)}, len(valset): {len(validation_set)}, len(testset): {len(test_set)}")

        train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=batch_size_train, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    # Optional learning rates for generator/discriminator
    generator_learning_rate = trial.suggest_categorical("generator_learning_rate", trials_config["generator_learning_rate"]) if "generator_learning_rate" in trials_config else None
    discriminator_learning_rate = trial.suggest_categorical("discriminator_learning_rate", trials_config["discriminator_learning_rate"]) if "discriminator_learning_rate" in trials_config else None

    # Update training config from trials_config if present
    for key in ["accumulate_grad_batches", "use_checkpoint", "gradient_clip", "freeze_generator", "disc_update_interval"]:
        if key in trials_config:
            config[key] = trials_config[key]

    # --- Model initialization ---
    model_config = trial.suggest_categorical("model_config", trials_config["model_config"]) if "model_config" in trials_config else {}
    model_class_name = trials_config.get("model_class_name", "VQGAN")
    model_class = {"VQGAN": VQGAN}[model_class_name]

    default_root_dir = output_dir

    if model_class_name == "MedicalDiffusionVQGAN":
        from types import SimpleNamespace
        model_config["default_root_dir"] = default_root_dir
        cfg = SimpleNamespace(**{"model": SimpleNamespace(**model_config)})
        vqgan_model = model_class(cfg)
    else:
        # Pass hyperparameters into model config
        if generator_learning_rate is not None:
            model_config["generator_learning_rate"] = generator_learning_rate
        if discriminator_learning_rate is not None:
            model_config["discriminator_learning_rate"] = discriminator_learning_rate
        for key in ["accumulate_grad_batches", "use_checkpoint", "gradient_clip", "freeze_generator", "disc_update_interval"]:
            if key in config:
                model_config[key] = config[key]

        vqgan_model = model_class(**model_config)
        vqgan_model.learning_rate = lr

    trial.set_user_attr("model_config", model_config)
    trial.set_user_attr("trials_config", trials_config)

    # --- Logging setup ---
    save_top_k = trials_config.get("save_top_k", 3)
    csv_logger = CSVLogger(default_root_dir, name="logger")
    loggers = [csv_logger]

    # --- Checkpoint directories ---
    checkpoints_dir = os.path.join(csv_logger.log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    train_dir = os.path.join(checkpoints_dir, "train")
    val_dir = os.path.join(checkpoints_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Training and validation checkpoint callbacks
    train_checkpoint = ModelCheckpoint(
        monitor="train/generator_total_loss", mode="min",
        save_top_k=save_top_k, dirpath=train_dir, save_last=True,
        filename="train-best-{epoch:02d}-{train_generator_total_loss:.4f}"
    )
    val_checkpoint = ModelCheckpoint(
        monitor="val/generator_total_loss", mode="min",
        save_top_k=save_top_k, dirpath=val_dir, save_last=True,
        filename="val-best-{epoch:02d}-{val_generator_total_loss:.4f}"
    )
    callbacks = [train_checkpoint, val_checkpoint]

    # Add optional extra callbacks
    if "callbacks" in trials_config:
        cb_cfg = trials_config["callbacks"]
        if "Checkpoints_every_5" in cb_cfg:
            every5_dir = os.path.join(checkpoints_dir, "checkpoints_every5")
            os.makedirs(every5_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(save_top_k=-1, every_n_epochs=5, dirpath=every5_dir, filename="epoch{epoch}"))
        if "Checkpoints_every_epoch" in cb_cfg:
            every_epoch_dir = os.path.join(checkpoints_dir, "checkpoints_every_epoch")
            os.makedirs(every_epoch_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(save_top_k=-1, every_n_epochs=1, dirpath=every_epoch_dir, filename="epoch{epoch}"))
        if "LPIPS_alex_metric" in cb_cfg:
            callbacks.append(LPIPSTopNModels3D(val_loader=validation_loader, net='alex', top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "LPIPS_alex_metric")))
        if "LPIPS_vgg_metric" in cb_cfg:
            callbacks.append(LPIPSTopNModels3D(val_loader=validation_loader, net='vgg', top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "LPIPS_vgg_metric")))
        if "MSSSIM_metric" in cb_cfg:
            callbacks.append(MSSSIMTopNModels3D(val_loader=validation_loader, top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "MSSSIM_metric")))
        if "FID_metric" in cb_cfg:
            callbacks.append(FIDTopNModels3D(val_loader=validation_loader, top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "FID_metric")))
        if "multimetric" in cb_cfg:
            callbacks.append(MultiMetricTopNModels3D(val_loader=validation_loader, top_n=3, dirpath=os.path.join(checkpoints_dir, "multimetric")))

    # --- Profiler setup ---
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_profiler = PyTorchProfiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./log_dir"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    # --- MLflow logging ---
    mlf_logger = mlf.get_logger()
    if mlf_logger is not None:
        mlf_logger.log_hyperparams(config)
        mlf_logger.log_hyperparams({
            "lr": lr,
            "generator_learning_rate": generator_learning_rate,
            "discriminator_learning_rate": discriminator_learning_rate,
            "model_config": model_config,
            "precision": trials_config["precision"]
        })
        loggers.append(mlf_logger)

    # --- Trainer setup ---
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        callbacks=callbacks,
        logger=loggers,
        max_epochs=config["num_epochs"],
        precision=trials_config["precision"],
        accelerator=accelerator,
        strategy="auto",
        profiler=my_profiler
    )

    # --- Training ---
    model_file_path = trials_config.get("model_file_path", None)

    if config.get("freeze_generator") and model_file_path:
        vqgan_model.load_pretrained_generator_only(model_file_path)
        trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    elif model_file_path is not None:
        if trials_config.get("clear_discriminator", False):
            checkpoint = torch.load(model_file_path, map_location="cpu")
            state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "discriminator" not in k}
            vqgan_model.load_state_dict(state_dict, strict=False)
            trainer.fit(vqgan_model, train_loader, validation_loader)
        else:
            trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader, ckpt_path=model_file_path)
    else:
        trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # --- Return best loss ---
    if "val/generator_total_loss" in trainer.callback_metrics:
        loss = trainer.callback_metrics["val/generator_total_loss"].item()
    else:
        loss = trainer.callback_metrics["train/generator_total_loss"].item()

    # Log artifacts
    checkpoint_dir = os.path.join(default_root_dir, "lightning_logs")
    if os.path.exists(checkpoint_dir):
        mlf.log_artifacts(checkpoint_dir, artifact_path="checkpoints")

    return loss


# =========================================================
# Helper: Load YAML config
# =========================================================
def load_trials_config(path_to_config):
    """Load trials configuration from YAML file."""
    with open(path_to_config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# =========================================================
# Main entrypoint
# =========================================================
if __name__ == '__main__':
    # --- CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_config_path', help='Path to trials config YAML')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", action='store_true', help="Use CUDA")
    parser.add_argument("-a", "--append", action='store_true', help="Append models to existing output dir")
    parser.add_argument("-m", "--mlflow", action='store_true', help="Use MLflow for experiment tracking")
    args = parser.parse_args(sys.argv[1:])

    # --- Basic paths & config ---
    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    config = {
        "num_epochs": trials_config["num_epochs"],
        "batch_size_train": trials_config["batch_size_train"],
        "batch_size_test": trials_config.get("batch_size_test", 1),
        "n_train_samples": trials_config.get("n_train_samples"),
        "n_test_samples": trials_config.get("n_test_samples"),
        "train_samples_ratio": trials_config.get("train_samples_ratio", 0.9),
        "val_samples_ratio": trials_config.get("val_samples_ratio", 0.2),
        "seed": trials_config.get("random_seed", 12345),
        "output_dir": output_dir,
    }

    # MLflow wrapper
    mlf = MLflowWrapper(args.mlflow)
    if args.mlflow:
        mlflow_config = trials_config["mlflow_config"]
        assert "tracking_uri" in mlflow_config, "MLFlow tracking URI missing"
        mlf.set_tracking_uri(mlflow_config["tracking_uri"])
        mlf.set_experiment(mlflow_config["experiment_name"])

    # Add optional transforms
    for key in ["input_transform", "output_transform", "output_iqr_scale", "normalize_input_indices", "normalize_output_indices", "data_file_name"]:
        if key in trials_config:
            config[key] = trials_config[key]

    # --- Reproducibility setup ---
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    pl.seed_everything(random_seed)

    # Handle output dir (clear or append)
    output_dir = os.path.join(output_dir, f"seed_{random_seed}")
    if os.path.exists(output_dir) and not args.append:
        #shutil.rmtree(output_dir)
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError(f"Output dir {output_dir} not found")

    # --- Optuna study setup ---
    sampler = TPESampler(seed=random_seed)
    if trials_config.get("sampler_class") == "BruteForceSampler":
        sampler = BruteForceSampler(seed=random_seed)

    study = optuna.create_study(sampler=sampler, direction="minimize")

    # --- Dataset preparation ---
    train_loader, validation_loader = None, None
    if not isinstance(config.get("n_train_samples"), (list, np.ndarray)):
        dset = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
        data_loader = DataLoader(dset, batch_size=config["batch_size_train"], shuffle=True)

    # --- Run Optuna optimization ---
    def obj_func(trial):
        return objective(trial, trials_config, train_loader, validation_loader)

    study.optimize(obj_func, n_trials=trials_config["num_trials"])

    # --- Report results ---
    pruned_trials = study.get_trials(states=[TrialState.PRUNED])
    complete_trials = study.get_trials(states=[TrialState.COMPLETE])
    print("\nStudy statistics:")
    print(f"  Finished trials: {len(study.trials)}")
    print(f"  Pruned trials:   {len(pruned_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    print("\nBest trial:")
    print(f"  Value: {best_trial.value}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # Save Optuna results
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)
    df = df.loc[df['state'] == 'COMPLETE'].drop('state', axis=1).sort_values('value')
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)
    print("\nOverall Results:\n", df)

    # Try to compute hyperparameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nMost important hyperparameters:")
        for k, v in importances.items():
            print(f"  {k}:{' ' * (15 - len(k))}{v * 100:.2f}%")
    except Exception as e:
        print(str(e))

    # Save study object
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))
