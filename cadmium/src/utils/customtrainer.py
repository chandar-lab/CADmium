from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import SequentialSampler
from transformers.trainer_utils import seed_worker, has_length
from transformers.utils import is_datasets_available
from typing import Optional
import torch
import datasets


class OrderedSFTTrainer(SFTTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            # Keep original length-grouping logic but disable its internal shuffling
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        # Modified core change here - use SequentialSampler instead of RandomSampler
        return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        dl = super().get_train_dataloader()
        # Force disable shuffling at DataLoader level
        dl.sampler.shuffle = False  
        dl.batch_sampler.shuffle = False
        return dl