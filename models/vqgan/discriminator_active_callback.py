from pytorch_lightning.callbacks import Callback
import os


class DiscriminatorActiveCheckpoint(Callback):
    def __init__(self, monitor='train/generator_total_loss', disc_start=3120, disc_ramp_duration=1000, save_top_k=3, dirpath='disc_checkpoints'):
        super().__init__()
        self.monitor = monitor
        self.threshold_step = disc_start + disc_ramp_duration
        self.save_top_k = save_top_k
        self.saved_checkpoints = []
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        current_step = trainer.global_step
        print("current step ", current_step)
        print(self.threshold_step)
        if current_step < self.threshold_step:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        filepath = os.path.join(
            self.dirpath,
            f"discactive-epoch{trainer.current_epoch:02d}-step{current_step}-loss{current_score:.4f}.ckpt"
        )
        trainer.save_checkpoint(filepath)
        self.saved_checkpoints.append((filepath, current_score.item()))

        # Sort and keep top K
        self.saved_checkpoints.sort(key=lambda x: x[1])
        if len(self.saved_checkpoints) > self.save_top_k:
            worst_ckpt = self.saved_checkpoints.pop(-1)
            try:
                os.remove(worst_ckpt[0])
            except OSError:
                pass
