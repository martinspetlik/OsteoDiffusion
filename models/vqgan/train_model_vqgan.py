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
from pytorch_lightning.loggers import CSVLogger
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.callbacks import Timer
#from models.cnn_diffusion.synthetic_CT_Unet import SyntheticCTUNet

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def objective(trial, trials_config, train_loader, validation_loader):
    best_vloss = 1_000_000.
    best_epoch = 0
    save_model_best_epoch = 0
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    batch_size_sample = trial.suggest_categorical("batch_size_sample", trials_config["batch_size_sample"])
    config["batch_size_train"] = batch_size_train
    config["batch_size_sample"] = batch_size_sample

    ####
    # Noise scheduler config
    ####
    beta_scheduler_type = "linear"
    scheduler_kwargs = {}
    if "beta_scheduler_type" in trials_config:
        beta_scheduler_type = trials_config["beta_scheduler_type"]
    if "scheduler_kwargs" in trials_config:
        scheduler_kwargs = trials_config["scheduler_kwargs"]

    loss_function = ["MSE", []]
    if "loss_function" in trials_config:
        loss_function = trial.suggest_categorical("loss_function", trials_config["loss_function"])

    loss_fn = get_loss_fn(loss_function)

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        # if "sub_datasets" in config and len(config["sub_datasets"]) > 0:
        #     train_set, validation_set, test_set = prepare_sub_datasets(study, config, data_dir=data_dir,
        #                                                                serialize_path=output_dir)
        # else:

        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set), len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    optimizer_name = "AdamW"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])

    if "mask_loss" in trials_config:
        config["mask_loss"] = trials_config["mask_loss"]
    if "weighted_mask_loss" in trials_config:
        config["weighted_mask_loss"] = trials_config["weighted_mask_loss"]
    if "accumulate_grad_batches" in trials_config:
        config["accumulate_grad_batches"] = trials_config["accumulate_grad_batches"]
    if "use_checkpoint" in trials_config:
        config["use_checkpoint"] = trials_config["use_checkpoint"]

    #####################
    #####################
    #  Initilize model #
    ####################
    ####################

    ##################################
    ### Graph neural network model ###
    ##################################
    model_config = {}
    if "model_config" in trials_config:
        model_config = trial.suggest_categorical("model_config", trials_config["model_config"])

    #num_node_attrs = trials_config["num_node_attrs"]

    ####
    ## SimpleUNet
    ####
    model_class_name = "SimpleUNet"
    if "model_class_name" in trials_config:
        model_class_name = trials_config["model_class_name"]

    # if model_class_name == "SimpleUNet":
    #      model_class = SimpleUNet
    # elif model_class_name == "UNet":
    #     model_class = UNet
    # elif model_class_name == "MedicalDiffusionUNet3D":
    #     model_class = MedicalDiffusionUNet3D
    # elif model_class_name == "MedicalDiffusionUNet3DOwn":
    #     model_class = MedicalDiffusionUNet3DOwn
    # elif model_class_name == "SyntheticCTUNet":
    #     model_class = SyntheticCTUNet
    cnn_model_class = MedicalDiffusionUNet3D
    if model_class_name == "MedicalDiffusionVQGAN":
        model_class = MedicalDiffusionVQGAN
    if model_class_name == "ALDMVQGAN":
        model_class = ALDMVQGAN
    if model_class_name == "VQGAN":
        model_class = VQGAN

    #cnn_kwargs = {'dim': 32, 'channels': 1}
    #cnn_model = UNet(**cnn_kwargs)
    #cnn_model = UNet3DMedicalDiffusion(**cnn_kwargs)
    print("model_config ", model_config)

    default_root_dir = output_dir

    print("model class", model_class)
    if model_class_name == "MedicalDiffusionVQGAN":
        from types import SimpleNamespace
        model_config["default_root_dir"] = default_root_dir
        cfg = SimpleNamespace(**{"model": SimpleNamespace(**model_config)})
        vqgan_model = model_class(cfg)

    if model_class_name == "ALDMVQGAN":
        vqgan_model = model_class(**model_config)
        vqgan_model.learning_rate = lr
        #cnn_model = cnn_model_class(**cnn_config)
    if model_class_name == "VQGAN":
        vqgan_model = model_class(**model_config)
        vqgan_model.learning_rate = lr
        if "accumulate_grad_batches" in config:
            vqgan_model.accumulate_grad_batches = config["accumulate_grad_batches"]
        if "use_checkpoint" in config:
            vqgan_model.use_checkpoint = config["use_checkpoint"]
            #cnn_model = cnn_model_class(**cnn_config)

    ####
    ## SimpleUNet
    ####
    #cnn_model = UNet(dim=32)
    #cnn_model =  UNet3DWithTimestep(in_channels=1, out_channels=1)
    #cnn_model = UNet3DAmir(in_channels=1, out_channels=1)

    # #######################
    # ### Noise scheduler ###
    # #######################
    # noise_scheduler_kwargs = {'beta_scheduler_type':beta_scheduler_type,
    #                           'num_timesteps':trials_config["num_timesteps"],
    #                           'scheduler_kwargs': scheduler_kwargs}
    # noise_scheduler = NoiseScheduler(**noise_scheduler_kwargs)

    #######################
    ### Diffusion model ###
    #######################
    #diff_model = DiffusionModel(cnn_model, vqgan_model, noise_scheduler).to(device)


    #########################
    #########################
    #########################

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr, "weight_decay": 0}
    #non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
    optimizer = None
    #print("optimizer kwargs ", optimizer_kwargs)

    #print("non frozen parameters ", non_frozen_parameters)
    # if len(non_frozen_parameters) > 0:
    #     optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

    print("optimizer ", optimizer)

    # trial.set_user_attr("diff_model_class", diff_model.__class__)
    # trial.set_user_attr("diff_model_name", diff_model._name)
    # trial.set_user_attr("cnn_model_class", cnn_model.__class__)
    #trial.set_user_attr("cnn_model_name", cnn_model._name)
    trial.set_user_attr("model_config", model_config)
    #trial.set_user_attr("num_node_attrs", num_node_attrs)
    #trial.set_user_attr("gnn_model_kwargs", gnn_kwargs)
    #trial.set_user_attr("noise_scheduler_kwargs", noise_scheduler_kwargs)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("trials_config", trials_config)

    # Training of the model
    start_time = time.time()
    avg_loss_list = []
    avg_vloss_list = []
    avg_vloss_list = []
    avg_vloss, avg_loss = best_vloss, best_vloss
    best_epoch = 0
    model_state_dict = {}
    optimizer_state_dict = {}

    model_path = 'trial_{}_losses_model_{}'.format(trial.number, vqgan_model._name)
    # print("model path ", model_path)
    # if os.path.exists(model_path):
    #     print("model pat hexists")
    #     return avg_vloss

    scheduler = None
    train = trials_config["train"] if "train" in trials_config else True

    # if "scheduler" in trials_config and optimizer is not None:
    #     trial_scheduler = trial.suggest_categorical("scheduler", trials_config["scheduler"])
    #     print("scheduler patience: {}, factor: {}".format(trial_scheduler["patience"], trial_scheduler["factor"]))
    #     if "class" in trial_scheduler:
    #         if trial_scheduler["class"] == "ReduceLROnPlateau":
    #             scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
    #                                                        patience=trial_scheduler["patience"],
    #                                                        factor=trial_scheduler["factor"])
    #         else:
    #             scheduler = lr_scheduler.StepLR(optimizer, step_size=trial_scheduler["step_size"],
    #                                         gamma=trial_scheduler["gamma"])

    # config = {
    #     "latent_dim": trial.suggest_int("latent_dim", 64, 256),
    #     "img_dim": 784,  # Example for MNIST
    #     "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
    #     "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    #     "num_epochs": trial.suggest_int("num_epochs", 10, 50)
    # }

    #model = VQGAN(config)
    from pytorch_lightning.callbacks import ModelCheckpoint

    print("trial ", trial)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    #callbacks.append(PyTorchLightningPruningCallback(trial, monitor='val/recon_loss'))
    #callbacks.append(Timer())
    #callbacks.append(CSVLogger(save_dir=os.path.join(default_root_dir, 'lightning_logs'), name="vqgan_model"))
    callbacks.append(ImageLogger(
         batch_frequency=750, max_images=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    accelerator = 'cpu'
    if torch.cuda.is_available():
        accelerator = 'cuda'

    from pytorch_lightning.profilers import PyTorchProfiler
    from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
    import pytorch_lightning as pl

    my_profiler = PyTorchProfiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./log_dir"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    csv_logger = CSVLogger(default_root_dir, name="logger")

    trainer = pl.Trainer(
        #gpus=cfg.model.gpus,
        #accumulate_grad_batches=trials_config["accumulate_grad_batches"],
        default_root_dir=default_root_dir,
        callbacks=callbacks,
        logger=[csv_logger],
        #max_steps=trials_config["max_steps"],
        max_epochs=config["num_epochs"], #trials_config["max_epochs"],
        precision=trials_config["precision"],
        #gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
        strategy="auto",
        #amp_backend="native",
        profiler=my_profiler
    )

    print("trainer.logger ", trainer.logger)

    trainer.fit(vqgan_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # trainer = pl.Trainer(max_epochs=config["num_epochs"], gpus=1 if torch.cuda.is_available() else 0)
    # trainer.fit(vqgan_model, train_dataloaders=train_loader)

    print("trainer.callback_metrics ", trainer.callback_metrics)

    print("total training time: ", time.time()- start_time)

    # Retrieve best loss
    return trainer.callback_metrics["val/recon_loss"].item()


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

    args = parser.parse_args(sys.argv[1:])

    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    config = {
              "num_epochs": trials_config["num_epochs"],
              "batch_size_train": trials_config["batch_size_train"],
              "batch_size_test": trials_config["batch_size_test"] if "batch_size_test" in trials_config else 250,
              "n_train_samples": trials_config["n_train_samples"] if "n_train_samples" in trials_config else None,
              "n_test_samples": trials_config["n_test_samples"] if "n_test_samples" in trials_config else None,
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.9,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "print_batches": 10,
              "init_norm": trials_config["init_norm"] if "init_norm" in trials_config else False,
              "init_norm_use_all_features": trials_config["init_norm_use_all_features"] if "init_norm_use_all_features" in trials_config else False,
              "log_all_input_channels": trials_config["log_all_input_channels"] if "log_all_input_channels" in trials_config else False,
              "log_input": trials_config["log_input"] if "log_input" in trials_config else True,
              "normalize_input": trials_config["normalize_input"] if "normalize_input" in trials_config else True,
              "log_output": trials_config["log_output"] if "log_output" in trials_config else False,
              "log10_output": trials_config["log10_output"] if "log10_output" in trials_config else False,
              "log_all_output": trials_config["log_all_output"] if "log_all_output" in trials_config else False,
              "log10_all_output": trials_config["log10_all_output"] if "log10_all_output" in trials_config else False,
              "normalize_output": trials_config["normalize_output"] if "normalize_output" in trials_config else True,
              "input_channels": trials_config["input_channels"] if "input_channels" in trials_config else None,
              "output_channels": trials_config["output_channels"] if "output_channels" in trials_config else None,
              "fractures_sep": trials_config["fractures_sep"] if "fractures_sep" in trials_config else False,
              "cross_section": trials_config["cross_section"] if "cross_section" in trials_config else False,
              "seed": trials_config["random_seed"] if "random_seed" in trials_config else 12345,
              "output_dir": output_dir,
              "sub_datasets": trials_config["sub_datasets"] if "sub_datasets" in trials_config else {}
              }

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
        dataset = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
        data_loader = DataLoader(dataset, batch_size=config["batch_size_train"], shuffle=True)

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

