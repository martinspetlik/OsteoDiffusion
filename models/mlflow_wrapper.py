from contextlib import contextmanager


class MLflowWrapper:
    def __init__(self, enabled):
        self.enabled = enabled
        self.tracking_uri = None
        self.experiment_name = ""
        if enabled:
            import mlflow
            self.mlflow = mlflow
        else:
            self.mlflow = None

    def start_run(self, **kwargs):
        if self.enabled:
            return self.mlflow.start_run(**kwargs)
        else:
            @contextmanager
            def dummy_run():
                yield
            return dummy_run()

    def log_param(self, key, value):
        if self.enabled:
            self.mlflow.log_param(key, value)

    def log_params(self, params_dict):
        if self.enabled:
            self.mlflow.log_params(params_dict)

    def log_metric(self, key, value, step=None):
        if self.enabled:
            if step is not None:
                self.mlflow.log_metric(key, value, step=step)
            else:
                self.mlflow.log_metric(key, value)

    def log_artifacts(self, local_dir, artifact_path=None):
        if self.enabled:
            self.mlflow.log_artifacts(local_dir, artifact_path)

    def set_tag(self, key, value):
        if self.enabled:
            self.mlflow.set_tag(key, value)

    def set_tracking_uri(self, uri):
        if self.enabled:
            self.tracking_uri = uri
            self.mlflow.set_tracking_uri(uri)

    def set_experiment(self, name):
        if self.enabled:
            self.experiment_name = name
            self.mlflow.set_experiment(name)

    def get_logger(self, type="pytorch_lightning"):
        if not self.enabled:
            return None

        if type == "pytorch_lightning":
            from pytorch_lightning.loggers import MLFlowLogger
            assert self.tracking_uri is not None, "Tracking URI is None - cannot initialize logger"
            return MLFlowLogger(
                experiment_name=self.experiment_name,
                tracking_uri=self.tracking_uri,
            )