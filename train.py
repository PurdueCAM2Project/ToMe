from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch
import torch.nn
import torch.profiler
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
import time

### Python
import os
from typing import List
import argparse

### Lightning
import lightning as L

### TIMM
import timm
import timm.optim
from timm.data import create_loader

### ToMe Backend / Wrapper
import tome
from tome.utils import parse_r

### Arch imports
from arch.regvit import deit_small_register_patch16_224, deit_base_register_patch16_224, deit_small_distilled_register_patch16_224, deit_base_distilled_register_patch16_224

###
### Argument parser init script
###
def get_args() -> argparse.Namespace:
    ### Grab any commandline arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    ### Generic parameters
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--profile-output-dir', default=None)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--dry-run-config-generate', action='store_true')
    parser.add_argument('--dataset-root-dir', type=str, default=None)
    parser.add_argument('--bin-dir', type=str, default="bin/")
    parser.add_argument('--dataset', type=str, choices=['imagenet1k'], default='imagenet1k')
    parser.add_argument('--timm-model', type=str, default='deit_small_patch16_224')
    parser.add_argument('--r', type=int, default=12)
    parser.add_argument('--r-list', nargs='+', default=None)

    ### Train parameters
    parser.add_argument('--train-strategy', type=str, default='ddp')
    parser.add_argument('--train-precision', type=str, default='16-mixed')
    parser.add_argument('--train-epochs', type=int, default=32)
    parser.add_argument('--train-num-devices', type=int, default=1)
    parser.add_argument('--train-num-nodes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    return args

###
### Used for parsing commandline args that are comma separated (nice way to pass lists to argparse)
###
def csv_string_to_int_list( string : str ) -> List:
    list = string.split(',')
    int_list = [int(element) for element in list]
    return int_list

###
### Parse all the options in the config.json, then update an argparse namespace with them
###
def update_argparse_options_from_json_config(
    config_json_path: str, args: argparse.Namespace
) -> argparse.Namespace:
    ### Load json config from the path
    with open(config_json_path, "r") as filehandle:
        json_dict = json.load(filehandle)

    ### Update 'args' based on json_dict items
    args_dict = vars(args)
    args_dict.update(json_dict)

    ### Return a new argparse Namespace instance
    updated_args = argparse.Namespace(**args_dict)
    return updated_args

###
### Dump options to a .JSON file
###
def save_argparse_options_to_json_config(
    config_json_path: str, args: argparse.Namespace
) -> None:
    ### Convert args into a dict
    args_dict = vars(args)

    ### Dump to string
    args_dict_json_str = json.dumps(args_dict, indent=4)

    ### Save to file
    with open(config_json_path, "w") as filehandle:
        filehandle.write(args_dict_json_str)

###
### Filename generation
###
def generate_pth_filename( profile : str, *args ) -> str:
    format_string = '{}'.format(profile)
    ### Append as many args as you'd like
    for arg in args:
        format_string += '_{}'
    format_string += '.pth'

    return format_string.format( *args )

###
### Save Model Checkpoint
###
def checkpoint_model(
        args : argparse.Namespace,
        fabric : L.Fabric,
        model : torch.nn.Module,
        model_optim : torch.nn.Module,
        lr_scheduler : torch.nn.Module,
        checkpoint_idx : int,
) -> None: 
    ### Save via fabric
    fabric.save(
        path=os.path.join(
            args.profile_output_dir,
            generate_pth_filename(
                args.profile, "lightning_checkpoint_{}".format(checkpoint_idx), "state"
            ),
        ),
        state={
            "model": model,
            "args": args,
            "model_optim": model_optim,
            "lr_scheduler": lr_scheduler,
        },
    )

###
### "Main" Evaluation Function
###
def train(
    args : argparse.Namespace, 
    fabric : L.Fabric,
    model : torch.nn.Module, 
    dataloader : DataLoader, 
    ):

    ### Set model to evaluation mode
    model.train()

    ### Compute lr here 
    lr = args.lr if args.lr else args.batch_size / 512.0 * 0.001

    ### Create optimizer, loss functions
    optimizer = timm.optim.AdamW(
        params=model.parameters(),
        lr = lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=1e-6,
        T_max=len(dataloader)
    )

    ### Setup with Fabric
    optimizer = fabric.setup_optimizers(optimizer)

    ### Cross Entropy Loss
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    ### Distillation loss
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    ### Checkpointing
    checkpoint_idx = 1

    for epoch_index in range(args.train_epochs):
        ### Create tqdm object
        dataloader_object = tqdm(dataloader) if args.train_num_devices == 1 and args.train_num_nodes == 1 else dataloader

        for batch_index, (input, target) in enumerate(dataloader_object):
            ### Warmup
            if batch_index < 1:
                for _ in range(10):
                    model(input)
                print('train.py: Finished warming up')
                torch.cuda.synchronize()

            ### Zero grad
            optimizer.zero_grad()

            with torch.no_grad():
                r_temp = model.r
                model.r = 0
                teacher_output = model(input)
                model.r = r_temp

            ### Forward pass!
            tome_output = model(input)

            loss = 0.5 * cross_entropy_loss(tome_output, target) + 0.5 * kl_loss(torch.nn.functional.log_softmax(tome_output,dim=-1), torch.nn.functional.softmax(teacher_output, dim=-1))
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()
            
            ###Update progress bar
            if isinstance(dataloader_object, tqdm):
                dataloader_object.set_description("Avg. Current Batch Loss: {:.2f}".format(loss.item()), refresh=True)

        ### Save to disk
        checkpoint_model(args, fabric, model, optimizer, scheduler, checkpoint_idx)
        checkpoint_idx += 1

###
### Entry point
###
if __name__ == '__main__':
    ### Get commandline args
    args = get_args()

    ### Create path to json config whether we use it or not
    config_path = os.path.normpath(os.path.join('config/', args.profile + '.json'))
    
    ### Check whether we are doing a dry-run
    if args.dry_run_config_generate:
        save_argparse_options_to_json_config(config_path, args)
        print('train.py: Created default config, exiting')
        exit(0)

    ### Update r list
    #if args.r_list != "":
    #    args.r_list = csv_string_to_int_list(args.r_list)
    #    print('train.py: args.r_list type: {} and value {}'.format(type(args.r_list, args.r_list)))

    ### Update parameters from config
    args = update_argparse_options_from_json_config(config_path, args)

    ### Pre-process args before using in train code
    args.dataset_root_dir = os.path.normpath(args.dataset_root_dir)
    args.bin_dir = os.path.normpath(args.bin_dir)

    ### Create output dir
    output_dir = os.path.join(args.bin_dir, args.profile)
    args.profile_output_dir = os.path.normpath(output_dir) 

    ### Set float32 matmul precision in case we have tensor cores
    ### Though, I imagine Fabric handles this?
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    ### Create Fabric instance
    fabric = L.Fabric(
        accelerator='cuda',
        strategy=args.train_strategy,
        devices=args.train_num_devices,
        num_nodes=args.train_num_nodes,
        precision=args.train_precision,
    )
    fabric.launch()

    ### Load ImageNet1K
    imagenet1k_dataset  = ImageFolder( root=os.path.join( args.dataset_root_dir, "train") )
    dataloader          = create_loader( imagenet1k_dataset, (3,224,224), args.batch_size, re_prob=0.1, use_prefetcher=False, is_training=True, num_workers=args.num_workers, persistent_workers=True )

    ### Load TIMM model
    model = timm.create_model(model_name=args.timm_model, pretrained=True)
    if model is None:
        print('train.py: Incorrect --timm-model: {}'.format(args.timm_model))
        exit(1)
    
    ### Wrap with ToMe
    tome.patch.timm(model)
    model.r = args.r_list if not args.r else args.r
    print('train.py: ToMe r type and value:{} {}'.format( type(model.r), model.r ))

    ### Setup model and dataloader with Fabric
    model       = fabric.setup_module(model)
    dataloader  = fabric.setup_dataloaders(dataloader)
    
    ### Start training time
    start_time = time.time()

    ### Launch eval(...)
    train(
        args=args,
        fabric=fabric,
        model=model,
        dataloader=dataloader,
    )

    ### Get ending time and print
    end_time = time.time()
    print('train.py: Finished training: took {:.2f} hours'.format( (end_time - start_time) / 60.0 / 60.0))