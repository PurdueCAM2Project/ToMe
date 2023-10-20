from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch
import torch.nn
import torch.profiler
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

### Python
import os
from typing import List
import argparse

### Lightning
import lightning as L

### TIMM
import timm
from timm.data import create_loader

### ToMe Backend / Wrapper
import tome
from tome.utils import parse_r

### Import Utilities
from utils import generate_pth_filename, update_argparse_options_from_json_config, save_argparse_options_to_json_config, get_args

###
### "Main" Evaluation Function
###
def evaluate(
    args : argparse.Namespace, 
    fabric : L.Fabric,
    model : torch.nn.Module, 
    dataloader : DataLoader, 
    profiler : torch.profiler.profiler.profile, 
    ):

    ### Set model to evaluation mode
    model.eval()

    inference_time_running          = 0.0
    inference_time_recording_count  = 0.0
    inference_time_average          = 0.0

    acc_top1_running                = 0.0
    acc_top1_recording_count        = 0.0
    acc_top1_average                = 0.0

    ###
    ### Iterate over the patch_series 
    ### Turn off gradient
    ###
    with torch.no_grad():
        start_event             = torch.cuda.Event(enable_timing=True)
        end_event               = torch.cuda.Event(enable_timing=True)

        ### Create tqdm object
        dataloader_object = tqdm(dataloader)

        for batch_index, (input, target) in enumerate(dataloader_object):
            ### Warmup
            if batch_index < 1:
                for _ in range(25):
                    model(input)
                torch.cuda.synchronize()

            ### Record inference time, do a forawrd pass
            start_event.record()

            output = model(input)

            ### Record inference time
            end_event.record()
            torch.cuda.synchronize()

            ### Compute accuracy for the hell of it
            class_prediction    = torch.argmax( output, dim=-1 )
            correct_prediction  = class_prediction == target

            ### Cache inference time
            inference_time_running          += start_event.elapsed_time( end_event )
            inference_time_recording_count  += 1.0
            inference_time_average          = inference_time_running / inference_time_recording_count

            ### Cache accuracy
            acc_top1_running            += correct_prediction.sum(dim=0)
            acc_top1_recording_count    += input.shape[0]
            acc_top1_average            = 100.0 * acc_top1_running / acc_top1_recording_count
            
            ### If we are using a profiler - step
            if profiler is not None:
                profiler.step()

            ### Update progress bar
            dataloader_object.set_description("Avg. Running Latency (ms): {:.2f} | Avg. Running Accuracy (ms): {:.2f}".format(inference_time_average, acc_top1_average.item()), refresh=True)

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
        print('eval.py: Created default config, exiting')
        exit(0)

    ### Update parameters from config
    args = update_argparse_options_from_json_config(config_path, args)

    ### Set matmul precision to high
    torch.set_float32_matmul_precision('high')

    ### Create Fabric instance
    fabric = L.Fabric(
        accelerator='cuda',
        strategy='dp',
        devices=1,
        num_nodes=1,
        precision='32',
    )
    fabric.launch()

    ### Load ImageNet1K
    imagenet1k_dataset  = ImageFolder( root=os.path.join( args.dataset_root_dir, "val") )
    dataloader          = create_loader( imagenet1k_dataset, (3,224,224), args.batch_size, use_prefetcher=False, is_training=False, num_workers=args.num_workers, persistent_workers=True )

    ### Load TIMM model
    model               = timm.create_model(model_name=args.timm_model, pretrained=True)
    if model is None:
        print('eval.py: Incorrect --timm-model: {}'.format(args.timm_model))
        exit(1)

    if args.resume_checkpoint_idx is not None:
        lightning_state = fabric.load(
            path=os.path.join(
                args.profile_output_dir,
                generate_pth_filename(
                    args.profile,
                    "lightning_checkpoint_{}".format(args.resume_checkpoint_idx),
                    "state",
                ),
            ),
        )
        model.load_state_dict(lightning_state["model"])
    
    ### Wrap with ToMe
    if args.no_wrap:
        pass
    else:
        tome.patch.timm(model)
        model.r = args.r_list if not args.r else args.r
        print('train.py: ToMe r type and value:{} / {}'.format( type(model.r), model.r ))

    model       = fabric.setup_module(model, move_to_device=True)
    dataloader  = fabric.setup_dataloaders(dataloader)
    
    ### Launch eval(...)
    evaluate(
        args=args,
        fabric=fabric,
        model=model,
        dataloader=dataloader,
        profiler=None,
    )