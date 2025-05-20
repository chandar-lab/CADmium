# utils/warmstabledecay.py
# ------------------------------------------------------------
# Warm-Stable-Decay learning-rate schedule **and**
# the callback that takes care of saving     warm/  and cold/
# checkpoints at the right steps.
# ------------------------------------------------------------
import os
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from transformers.modeling_utils import unwrap_model


class WarmStableDecayScheduler(LambdaLR):
    """
    LR profile:
        warm-up  : linearly 0 → 1   for `warmup_steps`
        stable   : 1                until `warm_ckpt_step`
        cool-down: 1 → min_lr_ratio over `decay_steps`
    """
    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        warm_ckpt_step: int,
        decay_steps: int,
        min_lr_ratio: float = 0.05,
        last_epoch: int = -1,
    ):
        assert warmup_steps <= warm_ckpt_step, (
            "warmup_steps must be ≤ warm checkpoint step (save_steps)")
        assert decay_steps > 0, "decay_steps must be positive"

        def lr_lambda(step: int):
            # --- warm-up ---
            if step < warmup_steps:
                return step / float(max(1, warmup_steps))

            # --- stable / flat ---
            if step < warm_ckpt_step:
                return 1.0

            # --- cool-down ---
            progress = step - warm_ckpt_step
            if progress >= decay_steps:
                return min_lr_ratio
            frac = 1.0 - progress / float(decay_steps)      # 1 → 0
            return min_lr_ratio + (1.0 - min_lr_ratio) * frac

        super().__init__(optimizer, lr_lambda, last_epoch)



# utils/warmstabledecay.py
import math, os, shutil, torch
from transformers import TrainerCallback, Trainer, TrainerControl

# ------------------------------------------------------------
# Hugging-Face  < 4.50  →  WEIGHTS_NAME in trainer_utils
# Hugging-Face ≥ 4.50   →  WEIGHTS_NAME in utils
# ------------------------------------------------------------
try:                                        # new location
    from transformers.utils import WEIGHTS_NAME
except ImportError:                         # fall back for old wheels
    from transformers.trainer_utils import WEIGHTS_NAME
import os, math, time
from transformers import TrainerCallback, TrainerControl
from transformers.trainer_utils import IntervalStrategy

class WarmStableDecayCallback(TrainerCallback):
    """
    At global_step == warm_step:        trigger a 'should_save' checkpoint in subdir warm/
    At global_step == cold_step: 
        - trigger a 'should_save' checkpoint in subdir cold/
        - trigger a full eval (control.should_evaluate)
        - trigger training stop
    """

    def __init__(self, *, warm_save: int, cold_save: int, trainer, verbose: bool = True):
        super().__init__()
        self.warm_save = warm_save
        self.cold_save = cold_save
        self.trainer  = trainer
        self.verbose  = verbose

    def _log(self, msg: str):
        if self.verbose and self.trainer.is_world_process_zero():
            t = time.strftime("%H:%M:%S", time.localtime())
            print(f"[WarmStableDecay {t}] {msg}", flush=True)

    def on_train_begin(self, args, state, control: TrainerControl, **kwargs):
        # Compute exactly when our warm & cold steps fall:
        # first checkpoint is after warmup+stable, aligned on resume
        gs = state.global_step

        self._log(f"Train begin at step {gs} → Will save warm ckpt at {self.warm_save} cold ckpt at {self.cold_save}")
        return control

    def on_step_end(self, args, state, control: TrainerControl, **kwargs):
        gs = state.global_step

        # always clear the default save flag
        control.should_save = False

        # Warm checkpoint
        if gs == self.warm_save:
            warm_dir = os.path.join(args.output_dir, "warm", f"checkpoint-{gs}")
            self._log(f"→ scheduling warm save at step {gs} to {warm_dir}")
            # tell Trainer: yes, please save right now
            control.should_save = True

        # Cold checkpoint + eval + stop
        if gs == self.cold_save:
            cold_dir = os.path.join(args.output_dir, "cold", f"checkpoint-{gs}")
            self._log(f"→ scheduling cold save at step {gs} to {cold_dir}")
            control.should_evaluate     = True
            control.should_save         = True
            control.should_training_stop = True

        return control