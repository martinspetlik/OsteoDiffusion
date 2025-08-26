import os
import sys
import argparse
import joblib
import torch
import pytorch_lightning as pl
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler
import time
import yaml
import shutil
import numpy as np
# #from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
from models.auxiliary_functions import get_loss_fn
# from metamodel.cnn.visualization.visualize_data import plot_samples, plot_dataset
from dataset.cnn_diffusion.dataset_preprocessing import prepare_dataset
from models.schedulers import NoiseScheduler
from torch.utils.data import DataLoader
from models.cnn_diffusion.medicaldiffusion_unet3D import MedicalDiffusionUNet3D
from models.vqgan.adopted_codes.medical_diffusion_vqgan import MedicalDiffusionVQGAN
from models.vqgan.adopted_codes.aldm_vqgan import ALDMVQGAN
from models.vqgan.vqgan_model import VQGAN
from models.vqgan.auxilliary_code import ImageLogger
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

import mlflow
from mlflow import log_params, log_metric
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.vqgan.discriminator_active_callback import DiscriminatorActiveCheckpoint
from models.vqgan.LPIPS_callback import LPIPSTopNModels3D
from models.vqgan.MSSSIM_callback import MSSSIMTopNModels3D
from models.vqgan.FID_callback import FIDTopNModels3D
from models.vqgan.multimetric_callback import MultiMetricTopNModels3D
from models.mlflow_wrapper import MLflowWrapper

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def objective(trial, trials_config, train_loader, validation_loader):
    # === Hyperparameter suggestions ===
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    batch_size_sample = trial.suggest_categorical("batch_size_sample", trials_config["batch_size_sample"])
    config["batch_size_train"] = batch_size_train
    config["batch_size_sample"] = batch_size_sample

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                            len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    generator_learning_rate = None
    if "generator_learning_rate" in trials_config:
        generator_learning_rate = trial.suggest_categorical("generator_learning_rate", trials_config["generator_learning_rate"])
    discriminator_learning_rate = None
    if "discriminator_learning_rate" in trials_config:
        discriminator_learning_rate = trial.suggest_categorical("discriminator_learning_rate", trials_config["discriminator_learning_rate"])

    if "accumulate_grad_batches" in trials_config:
        config["accumulate_grad_batches"] = trials_config["accumulate_grad_batches"]
    if "use_checkpoint" in trials_config:
        config["use_checkpoint"] = trials_config["use_checkpoint"]
    if "gradient_clip" in trials_config:
        config["gradient_clip"] = trials_config["gradient_clip"]
    if "freeze_generator" in trials_config:
        config["freeze_generator"] = trials_config["freeze_generator"]

    # === Model init ===
    model_config = {}
    if "model_config" in trials_config:
        model_config = trial.suggest_categorical("model_config", trials_config["model_config"])

    model_class_name = trials_config.get("model_class_name", "VQGAN")
    model_class = {"MedicalDiffusionVQGAN": MedicalDiffusionVQGAN, "ALDMVQGAN": ALDMVQGAN, "VQGAN": VQGAN}[model_class_name]

    default_root_dir = output_dir

    if model_class_name == "MedicalDiffusionVQGAN":
        from types import SimpleNamespace
        model_config["default_root_dir"] = default_root_dir
        cfg = SimpleNamespace(**{"model": SimpleNamespace(**model_config)})
        vqgan_model = model_class(cfg)
    else:
        if generator_learning_rate is not None:
            model_config["generator_learning_rate"] = generator_learning_rate
        if discriminator_learning_rate is not None:
            model_config["discriminator_learning_rate"] = discriminator_learning_rate
        if "accumulate_grad_batches" in config:
            model_config["accumulate_grad_batches"] = config["accumulate_grad_batches"]
        if "use_checkpoint" in config:
            model_config["use_checkpoint"] = config["use_checkpoint"]
        if "gradient_clip" in config:
            model_config["gradient_clip"] = config["gradient_clip"]
        if "freeze_generator" in config:
            model_config["freeze_generator"] = config["freeze_generator"]
        vqgan_model = model_class(**model_config)
        vqgan_model.learning_rate = lr

    trial.set_user_attr("model_config", model_config)
    trial.set_user_attr("trials_config", trials_config)

    ####################
    # Start MLflow Run #
    ####################
    # with mlf.start_run(nested=True):
    #     mlf.set_tag("model_class", model_class_name)
    #     mlf.log_params(config)
        # mlf.log_params({
        #     "lr": lr,
        #     "batch_size_train": batch_size_train,
        #     "batch_size_sample": batch_size_sample,
        #     "loss_function": loss_function[0] if isinstance(loss_function, list) else str(loss_function),
        #     "model_config": model_config,
        #     "accumulate_grad_batches": config.get("accumulate_grad_batches", None),
        #     "use_checkpoint": config.get("use_checkpoint", None),
        #     "precision": trials_config["precision"],
        #     "random_seed": trials_config["random_seed"]
        # })

    save_top_k = trials_config.get("save_top_k", 3)

    csv_logger = CSVLogger(default_root_dir, name="logger")
    loggers = [csv_logger]

    checkpoints_dir = os.path.join(csv_logger.log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Paths for checkpoints
    train_dir = os.path.join(checkpoints_dir, "train")
    val_dir = os.path.join(checkpoints_dir, "val")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    checkpoint_filename = trials_config.get("model_checkpoint_filename", 'latest_checkpoint')

    train_checkpoint = ModelCheckpoint(
        monitor="train/generator_total_loss",
        save_top_k=save_top_k,
        mode="min",
        dirpath=train_dir, save_last=True,
        filename="train-best-{epoch:02d}-{train_generator_total_loss:.4f}"
    )

    val_checkpoint = ModelCheckpoint(
        monitor="val/generator_total_loss",
        save_top_k=save_top_k,
        mode="min",
        dirpath=val_dir, save_last=True,
        filename="val-best-{epoch:02d}-{val_generator_total_loss:.4f}"
    )

    callbacks = [train_checkpoint, val_checkpoint]

    if "callbacks" in trials_config:
        if "LPIPS_alex_metric" in trials_config["callbacks"]:
            LPIPS_alex_callback = LPIPSTopNModels3D(val_loader=validation_loader, net='alex', top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "LPIPS_alex_metric"))
            callbacks.append(LPIPS_alex_callback)
        if "LPIPS_vgg_metric" in trials_config["callbacks"]:
            LPIPS_vgg_callback = LPIPSTopNModels3D(val_loader=validation_loader, net='vgg', top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "LPIPS_vgg_metric"))
            callbacks.append(LPIPS_vgg_callback)
        if "MSSSIM_metric" in trials_config["callbacks"]:
            MSSSIM_callback = MSSSIMTopNModels3D(val_loader=validation_loader, top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "MSSSIM_metric"))
            callbacks.append(MSSSIM_callback)
        if "FID_metric" in trials_config["callbacks"]:
            FID_callback = FIDTopNModels3D(val_loader=validation_loader, top_n=save_top_k, dirpath=os.path.join(checkpoints_dir, "FID_metric"))
            callbacks.append(FID_callback)
        if "multimetric" in trials_config["callbacks"]:
            multimetric_callback = MultiMetricTopNModels3D(val_loader=validation_loader, top_n=3, dirpath=os.path.join(checkpoints_dir, "multimetric"))
            callbacks.append(multimetric_callback)

    # train_checkpoint = ModelCheckpoint(monitor='train/generator_total_loss',
    #                 save_top_k=3, mode='min', filename='latest_checkpoint'),
    #
    # val_checkpoint = ModelCheckpoint(monitor='val/generator_total_loss',
    #                                    save_top_k=3, mode='min', filename='latest_checkpoint'),

    # # Checkpoint for best training generator loss
    # train_checkpoint = ModelCheckpoint(
    #     monitor="train/generator_total_loss",
    #     save_top_k=save_top_k,  # or your save_top_k
    #     mode="min",
    #     filename="train-best-{epoch:02d}-{train/generator_total_loss:.4f}",
    #     save_last=True
    # )
    #
    # # Checkpoint for best validation generator loss
    # val_checkpoint = ModelCheckpoint(
    #     monitor="val/generator_total_loss",
    #     save_top_k=save_top_k,  # or your save_top_k
    #     mode="min",
    #     dirpath="checkpoints/",
    #     filename="val-best-{epoch:02d}-{val/generator_total_loss:.4f}",
    #     save_last=True  # optionally keep last validation checkpoint
    # )

    # disc_checkpoint_train = DiscriminatorActiveCheckpoint(
    #     monitor="train/generator_total_loss",
    #     disc_start=model_config["loss_config"].get("disc_start", 0),
    #     disc_ramp_duration=model_config["loss_config"].get("disc_ramp_duration", 0),
    #     save_top_k=save_top_k,
    #     #dirpath=csv_logger.log_dir,
    #     dirpath="checkpoints/",
    #     filename="train-disc-best-{epoch:02d}-{train/generator_total_loss:.4f}",
    #     mode="min"
    # )
    #
    # disc_checkpoint_val = DiscriminatorActiveCheckpoint(
    #     monitor="val/generator_total_loss",
    #     disc_start=model_config["loss_config"].get("disc_start", 0),
    #     disc_ramp_duration=model_config["loss_config"].get("disc_ramp_duration", 0),
    #     save_top_k=save_top_k,
    #     #dirpath=csv_logger.log_dir,
    #     dirpath="checkpoints/",
    #     filename="val-disc-best-{epoch:02d}-{val/generator_total_loss:.4f}",
    #     mode="min"
    # )

   #, disc_checkpoint_train, disc_checkpoint_val]

    # trainer = pl.Trainer(
    #     callbacks=[train_checkpoint, val_checkpoint],
    #     # other args...
    # )
    #
    # callbacks = [
    #     # Save best model based on validation generator loss
    #     ModelCheckpoint(
    #         monitor='val/generator_total_loss',
    #         save_top_k=save_top_k,
    #         mode='min',
    #         filename='val-best-{epoch:02d}-{val/generator_total_loss:.4f}',
    #         save_last=True
    #     ),
    #
    #     # Save best model based on training generator loss (for debugging)
    #     ModelCheckpoint(
    #         monitor='train/generator_total_loss',
    #         save_top_k=save_top_k,
    #         mode='min',
    #         filename='train-best-{epoch:02d}-{train/generator_total_loss:.4f}',
    #     ),
    #
    #     # Discriminator checkpoint based on validation generator loss
    # DiscriminatorActiveCheckpoint(
    #     monitor='val/generator_total_loss',
    #     disc_start=model_config["loss_config"].get("disc_start", 0),
    #     disc_ramp_duration=model_config["loss_config"].get("disc_ramp_duration", 0),
    #     save_top_k=save_top_k,
    #     dirpath=csv_logger.log_dir,
    #     filename='val-disc-best-{epoch:02d}-{val/generator_total_loss:.4f}',
    #     mode='min'
    # ),
    #
    #     # Discriminator checkpoint based on training generator loss (for debugging)
    #     DiscriminatorActiveCheckpoint(
    #         monitor='train/generator_total_loss',
    #         disc_start=model_config["loss_config"].get("disc_start", 0),
    #         disc_ramp_duration=model_config["loss_config"].get("disc_ramp_duration", 0),
    #         save_top_k=1,
    #         dirpath=csv_logger.log_dir,
    #         filename='train-disc-best-{epoch:02d}-{train/generator_total_loss:.4f}',
    #         mode='min'
    #     )
    # ]

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

    my_profiler = PyTorchProfiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./log_dir"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

    mlf_logger = mlf.get_logger()
    if mlf_logger is not None:
        mlf_logger.log_hyperparams(config)
        log_params_dict = {"lr": lr,
                           "generator_learning_rate": generator_learning_rate,
                           "discriminator_learning_rate": discriminator_learning_rate,
                           "model_config": model_config,
                           "precision": trials_config["precision"]}

        mlf_logger.log_hyperparams(log_params_dict)
        loggers.append(mlf_logger)

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

    model_file_path = trials_config.get("model_file_path", None)

    start_time = time.time()

    if "freeze_generator" in config and config["freeze_generator"] and model_file_path:
        vqgan_model.load_pretrained_generator_only(model_file_path)
        trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    elif model_file_path is not None:
        if "clear_discriminator" in trials_config and trials_config["clear_discriminator"]:
            checkpoint = torch.load(model_file_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            # Filter out all discriminator weights
            filtered_state_dict = {k: v for k, v in state_dict.items() if "discriminator" not in k}
            # Load only generator weights
            missing, unexpected = vqgan_model.load_state_dict(filtered_state_dict, strict=False)
            trainer.fit(vqgan_model, train_loader, validation_loader)
        else:
            trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader, ckpt_path=model_file_path)

    else:
        trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    if "val/generator_total_loss" in trainer.callback_metrics:
        loss = trainer.callback_metrics["val/generator_total_loss"].item()
    else:
        loss = trainer.callback_metrics["train/generator_total_loss"].item()
    #mlf.log_metric("train_generator_total_loss", loss)
    #mlf.log_metric("training_duration_sec", training_duration)

    # Optional: Log model checkpoint artifact (if available)
    checkpoint_dir = os.path.join(default_root_dir, "lightning_logs")
    if os.path.exists(checkpoint_dir):
        mlf.log_artifacts(checkpoint_dir, artifact_path="checkpoints")

    return loss


def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_config_path', help='Path tp trials config')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    parser.add_argument("-a", "--append", default=False, action='store_true', help="append models")
    parser.add_argument("-m", "--mlflow", default=False, action='store_true', help="use mlflow to track experiments")

    args = parser.parse_args(sys.argv[1:])

    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    config = {
              "num_epochs": trials_config["num_epochs"],
              "batch_size_train": trials_config["batch_size_train"],
              "batch_size_test": trials_config["batch_size_test"] if "batch_size_test" in trials_config else 1,
              "n_train_samples": trials_config["n_train_samples"] if "n_train_samples" in trials_config else None,
              "n_test_samples": trials_config["n_test_samples"] if "n_test_samples" in trials_config else None,
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.9,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "seed": trials_config["random_seed"] if "random_seed" in trials_config else 12345,
              "output_dir": output_dir,
              }

    mlf = MLflowWrapper(args.mlflow)

    if args.mlflow:
        mlflow_config = trials_config["mlflow_config"]
        assert "tracking_uri" in mlflow_config, "MLFlow tracking uri is missing in the configuration"
        mlf.set_tracking_uri(mlflow_config["tracking_uri"])
        mlf.set_experiment(mlflow_config["experiment_name"])

    if "input_transform" in trials_config:
        config["input_transform"] = trials_config["input_transform"]
    if "output_transform" in trials_config:
        config["output_transform"] = trials_config["output_transform"]
    if "output_iqr_scale" in trials_config:
        config["output_iqr_scale"] = trials_config["output_iqr_scale"]
    if "normalize_input_indices" in trials_config:
        config["normalize_input_indices"] = trials_config["normalize_input_indices"]
    if "normalize_output_indices" in trials_config:
        config["normalize_output_indices"] = trials_config["normalize_output_indices"]

    if "data_file_name" in trials_config:
        config["data_file_name"] = trials_config["data_file_name"]

    # Optuna params
    num_trials = trials_config["num_trials"]

    print("use cuda ", use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)
    print("config seed ", config["seed"])

    # Make runs repeatable
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    pl.seed_everything(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir) and not args.append:
        #shutil.rmtree(output_dir)
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError("output dir {} not exists".format(output_dir))

    sampler = TPESampler(seed=random_seed)
    if "sampler_class" in trials_config:
        if trials_config["sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)

    study = optuna.create_study(sampler=sampler, direction="minimize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_loader, validation_loader = None, None
    if "n_train_samples" not in config or not isinstance(config["n_train_samples"], (list, np.ndarray)):
        dset = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
        data_loader = DataLoader(dset, batch_size=config["batch_size_train"], shuffle=True)

    def obj_func(trial):
        return objective(trial, trials_config, train_loader, validation_loader)

    study.optimize(obj_func, n_trials=num_trials)

    # ================================
    # Results
    # ================================
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Display the most important hyperparameters
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    except Exception as e:
        print(str(e))

    # serialize optuna study object
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))

