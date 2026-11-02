"""Convert genuinely constrained successful trajectories to five-action learning targets."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robomimic'))
from action_adapter import KEEP,assert_constrained7,reduce_action7


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def convert_training_data(source,output,count=1000):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if count<=0:raise ValueError('Requested training demo count must be positive')
    source_hash=checksum(source)
    with h5py.File(source,'r') as raw:
        names=sorted(raw['data'],key=lambda s:int(s.split('_')[-1]))
        if len(names)<count:raise ValueError(f'Need {count} successes, found {len(names)}')
        for name in names:
            episode=raw['data'][name]
            if not bool(episode.attrs.get('success',False)):raise ValueError(f'Failed trajectory in success dataset: {name}')
            actions=episode['actions'][()]
            assert_constrained7(actions)
            if actions.ndim!=2 or len(actions)<=0 or len(actions)!=int(episode.attrs['num_samples']):
                raise ValueError(f'Invalid trajectory length: {name}')
            def finite(key,obj):
                if isinstance(obj,h5py.Dataset) and np.issubdtype(obj.dtype,np.number) and not np.isfinite(obj[()]).all():
                    raise ValueError(f'Nonfinite trajectory data: {name}/{key}')
            episode.visititems(finite)
        lengths=[];created=False
        try:
            with h5py.File(output,'x') as target:
                created=True
                for key,value in raw.attrs.items():target.attrs[key]=value
                target.attrs['policy_action_dimension']=5
                target.attrs['environment_action_dimension']=7
                target.attrs['policy_action_indices']=np.array(KEEP,dtype=np.int32)
                target.attrs['source_constrained_dataset']=str(source)
                data=target.create_group('data')
                for key,value in raw['data'].attrs.items():data.attrs[key]=value
                for index,name in enumerate(names[:count]):
                    raw.copy(raw['data'][name],data,name=f'demo_{index}')
                    ep=data[f'demo_{index}'];old=ep['actions'];attrs=dict(old.attrs)
                    short=reduce_action7(old[()]);del ep['actions']
                    reduced=ep.create_dataset('actions',data=short)
                    for key,value in attrs.items():reduced.attrs[key]=value
                    lengths.append(len(short))
                data.attrs['total']=sum(lengths)
            if checksum(source)!=source_hash:raise ValueError('Raw generation source changed during conversion')
        except BaseException:
            if created:output.unlink(missing_ok=True)
            raise
    return {'raw_successes':len(names),'training_demos':count,'total_training_steps':sum(lengths),
            'length_min_mean_max':[min(lengths),float(np.mean(lengths)),max(lengths)],
            'training_dataset':str(output),'training_dataset_sha256':checksum(output),
            'constrained_raw_dataset':str(source),'constrained_raw_sha256':source_hash,
            'policy_action_dimension':5,'environment_action_dimension':7,'kept_action_indices':KEEP,
            'all_recorded_removed_rotation_commands_exactly_zero':True,
            'all_finite':True,'all_success_labels':True,'observations_preserved':True}


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(convert_training_data(args.input_file, args.output_file, args.count), indent=2))
