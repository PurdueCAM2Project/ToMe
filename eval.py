import torch
import torch.nn
import torch.profiler
import torch.utils.data
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from typing import List
import os
import lightning as L
import timm
###
### Argument parser init script
###
def get_args() -> argparse.Namespace:
    ### Grab any commandline arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    ### Config or manual entry
    parser.add_argument('--dataset', type=str, choices=['imagenet1k'])
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--tensorboard-profiling', action='store_true')
    parser.add_argument('--arch', type=str, choices=['deit_small_patch16_224'])
    parser.add_argument('--resume-model', type=str)
    args = parser.parse_args()

    return args

###
### "Main" Evaluation Function
###
def evaluate(
    args : argparse.Namespace, 
    fabric : L.Fabric,
    model : torch.nn.Module, 
    dataloader : DataLoader, 
    profiler : torch.profiler.profiler.profile, 
    device : torch.device, 
    **kwargs):

    ### Set model to evaluation mode
    model.eval()

    inference_time_running          = 0.0
    inference_time_recording_count  = 0.0
    inference_time_average          = 0.0

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

            output = model.forward(input)

            ### Record inference time
            end_event.record()
            torch.cuda.synchronize()

            ### Cache inference time
            inference_time_running += start_event.elapsed_time( end_event )
            inference_time_recording_count += 1.0
            inference_time_average = inference_time_running / inference_time_recording_count
            
            ### If we are using a profiler - step
            if profiler is not None:
                profiler.step()

            ### Update progress bar
            dataloader_object.set_description("Avg. Running Latency (ms): {:.2f}".format(inference_time_average), refresh=True)

###
### Entry point
###
if __name__ == '__main__':
    pass