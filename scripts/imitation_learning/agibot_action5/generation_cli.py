"""Resolve effective production options before the upstream writer can overwrite a file."""
import argparse
from pathlib import Path


def generation_arguments(argv, pilot_attempts=None):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--output_file',type=Path,required=True)
    parser.add_argument('--num_envs',type=int,default=16)
    parser.add_argument('--generation_num_trials',type=int,default=None)
    options,remaining=parser.parse_known_args(argv)
    target=1000 if pilot_attempts is None else pilot_attempts
    if options.num_envs!=16:
        raise ValueError('This production protocol requires exactly 16 parallel environments')
    if options.generation_num_trials not in (None,target):
        raise ValueError(f'This protocol requires generation_num_trials={target}')
    # setup_output_paths strips the final suffix; the HDF5 writer appends .hdf5.
    output=options.output_file.resolve().with_suffix('.hdf5')
    failed=output.with_name(output.stem+'_failed.hdf5')
    for destination in (output,failed):
        if destination.exists():raise FileExistsError(destination)
    forwarded=[*remaining,'--output_file',str(output),'--num_envs','16','--generation_num_trials',str(target)]
    return forwarded,output
