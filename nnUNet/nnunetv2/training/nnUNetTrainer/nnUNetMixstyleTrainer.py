from .nnUNetTrainer import nnUNetTrainer
from nnunetv2.network_architecture.UNet_Mixstyle import get_network
import torch.nn as nn
from typing import Union, List, Tuple
class nnUNetMixstyleTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                device=None):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1000 
        self.print_to_log_file(f"Using nnUNetMixstyleTrainer, building network architecture directly, \nwithout using the setting from plans.json")
    
    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        include_identifiers = ['mca', 'nu','nyu','cad','iu','ahn','emc','mcf','northwestern']
        
        # only use the following datasets for training and validation
        dataset_tr.identifiers = [id for id in dataset_tr.identifiers 
                                  if any(incl in id.lower() for incl in include_identifiers)]
        dataset_val.identifiers = [id for id in dataset_val.identifiers 
                                   if any(incl in id.lower() for incl in include_identifiers)]
        
        print("You are using data including the following datasets: ", include_identifiers)
        print("Training dataset size: ", len(dataset_tr.identifiers))
        print("Validation dataset size: ", len(dataset_val.identifiers))
        return dataset_tr, dataset_val  


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network()