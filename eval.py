from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch
import torch.nn
import torch.profiler
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import warnings

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

    ### Get len of dataset
    batch_count = len(dataloader) if args.forward_pass_count is None else args.forward_pass_count

    ### Lists that we use to collect data about our model
    ### NOTE: For timing, we put the tensors on the CPU since the CUDA timing returns Python float scalars
    ### NOTE: Check whether we are enforcing a forward pass count
    model_inference_time_tensor = torch.zeros(
        size=(batch_count,), dtype=torch.float32, device="cpu"
    )
    model_correct_prediction_tensor = torch.zeros(
        size=(batch_count,), dtype=torch.float32, device="cuda"
    )
    model_prediction_count = 0

    ###
    ### Iterate over the patch_series 
    ### Turn off gradient
    ###
    with torch.no_grad():
        ### Create tqdm object
        dataloader_object = tqdm(dataloader)

        for batch_index, (input, target) in enumerate(dataloader_object):
            ### Create CUDA timing events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            ### Warmup
            if batch_index < 1:
                for _ in range(10):
                    model(input)
                torch.cuda.synchronize()

            ### Early exit
            if batch_index >= batch_count:
                break

            ### Record inference time, do a forawrd pass
            start_event.record()

            output = model(input)

            ### Record inference time
            end_event.record()
            torch.cuda.synchronize()

            ### Update prediction count based on batch size
            model_prediction_count += input.shape[0]

            ### Perform argmax to get the prediction
            class_correct_prediction = torch.argmax(output, dim=1) == target

            ### Cache inference time
            model_inference_time_tensor[batch_index] = start_event.elapsed_time(
                end_event
            )

            ### Cache correct prediction
            ### TODO: Add a way to parse the type of dataset we are working with - this works for classification tasks
            model_correct_prediction_tensor[batch_index] = class_correct_prediction.sum(
                dim=0
            )
            
            ### If we are using a profiler - step
            if profiler is not None:
                profiler.step()

            ### Update progress bar
            #dataloader_object.set_description("Avg. Running Latency (ms): {:.2f} | Avg. Running Accuracy (ms): {:.2f}".format(inference_time_average, acc_top1_average.item()), refresh=True)
            dataloader_object.set_description("Avg. Running Latency (ms): {:.2f} | Avg. Running Accuracy (%): {:.2f}".format(
                model_inference_time_tensor[0:(batch_index+1)].mean().item(), 
                100.0 * model_correct_prediction_tensor[0:(batch_index+1)].sum().item() / model_prediction_count, 
                refresh=True)
            )

def list_to_int_list( list : List) -> List[int]:
    return [int(k) for k in list]

###
### Entry point
###
if __name__ == '__main__':
    ### Ignore all warnings
    warnings.filterwarnings("ignore")

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
    if args.load_config:
        args = update_argparse_options_from_json_config(config_path, args)

    ### Set matmul precision to high
    torch.set_float32_matmul_precision('high')

    ### Create Fabric instance
    fabric = L.Fabric(
        accelerator='cuda',
        strategy='dp',
        devices=1,
        num_nodes=1,
        precision='32-true',
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
        model.r = list_to_int_list(args.r_list) if not args.r else args.r
        print('eval.py: ToMe r: {}'.format( model.r ))

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