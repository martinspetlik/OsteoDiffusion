from pytorch_lightning.callbacks import ModelCheckpoint


class DiscriminatorActiveCheckpoint(ModelCheckpoint):
    def __init__(self, disc_start: int = 0, disc_ramp_duration: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_start = disc_start
        self.disc_ramp_duration = disc_ramp_duration

    def _should_skip_saving(self, trainer) -> bool:
        """Return True if we should skip saving because discriminator is not active yet."""
        # Activate discriminator after ramp start + duration
        current_step = trainer.global_step
        active = current_step >= (self.disc_start + self.disc_ramp_duration)
        return not active

    def _save_checkpoint(self, trainer, filepath) -> None:
        # Skip saving until discriminator is active
        if self._should_skip_saving(trainer):
            return
        super()._save_checkpoint(trainer, filepath)
