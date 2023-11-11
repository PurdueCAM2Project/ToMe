import argparse
import json

###
### Argument parser init script
###
def get_args() -> argparse.Namespace:
    ### Grab any commandline arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, add_help=False)

    ### General Parameters
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--profile-output-dir', default=None)
    parser.add_argument('--load-config', action='store_true')
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--resume-checkpoint-idx', type=int, default=None)
    parser.add_argument('--dry-run-config-generate', action='store_true')
    parser.add_argument('--bin-dir', type=str, default="bin/")
    parser.add_argument('--timm-model', type=str, default='deit_small_patch16_224')
    parser.add_argument('--dataset', type=str, choices=['imagenet1k'])
    parser.add_argument('--dataset-root-dir', type=str, default=None)
    parser.add_argument('--tensorboard-profiling', action='store_true')
    parser.add_argument('--forward-pass-count', default=None)
    parser.add_argument('--no-progress-bar', action='store_true')

    ### ToMe Parameters
    parser.add_argument('--r', type=int, default=None)
    parser.add_argument('--r-list', nargs='+', default=[12]*12)
    parser.add_argument('--no-wrap', action='store_true')

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