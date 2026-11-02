import json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
try:
    import training_data as T
except ModuleNotFoundError:
    T=None

class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(T,'Constrained 7D to 5D training data conversion is not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)
        self.src=self.p/'raw.hdf5';self.dst=self.p/'train.hdf5'
        with h5py.File(self.src,'w') as f:
            f.attrs['origin']='actual-constrained-replay';data=f.create_group('data');data.attrs['env_args']=json.dumps({'env_name':'unchanged7d'})
            for i in range(2):
                g=data.create_group(f'demo_{i}');g.attrs['success']=True;g.attrs['num_samples']=2
                g.create_dataset('actions',data=np.array([[1,2,3,0,4,0,1],[5,6,7,0,8,0,-1]],dtype=np.float32))
                g.create_dataset('obs/eef_quat',data=np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32))
                g.create_dataset('states/robot',data=np.array([[11,12],[13,14]],dtype=np.float32))
            data.attrs['total']=4
    def test_conversion_removes_only_command_columns_and_preserves_observations(self):
        before=self.src.read_bytes();r=T.convert_training_data(self.src,self.dst,count=2)
        self.assertEqual(r['training_demos'],2);self.assertEqual(r['total_training_steps'],4)
        self.assertEqual(self.src.read_bytes(),before)
        with h5py.File(self.dst,'r') as f:
            np.testing.assert_array_equal(f['data/demo_0/actions'][()],[[1,2,3,4,1],[5,6,7,8,-1]])
            np.testing.assert_array_equal(f['data/demo_0/obs/eef_quat'][()],[[1,0,0,0],[0,1,0,0]])
            np.testing.assert_array_equal(f['data/demo_0/states/robot'][()],[[11,12],[13,14]])
            self.assertEqual(f['data'].attrs['total'],4);self.assertEqual(f.attrs['environment_action_dimension'],7)
            self.assertEqual(f.attrs['policy_action_dimension'],5)
    def test_nonzero_removed_rotation_is_rejected_without_creating_output(self):
        with h5py.File(self.src,'r+') as f:f['data/demo_1/actions'][0,5]=1e-9
        with self.assertRaises(ValueError):T.convert_training_data(self.src,self.dst,count=2)
        self.assertFalse(self.dst.exists())
    def test_unsuccessful_or_nonfinite_trajectory_cannot_enter_training(self):
        with h5py.File(self.src,'r+') as f:f['data/demo_1'].attrs['success']=False
        with self.assertRaises(ValueError):T.convert_training_data(self.src,self.dst,count=2)
        self.assertFalse(self.dst.exists())
        with h5py.File(self.src,'r+') as f:f['data/demo_1'].attrs['success']=True;f['data/demo_1/obs/eef_quat'][0,0]=np.nan
        with self.assertRaises(ValueError):T.convert_training_data(self.src,self.dst,count=2)
        self.assertFalse(self.dst.exists())
    def test_existing_output_and_insufficient_success_count_are_preserved(self):
        with self.assertRaises(ValueError):T.convert_training_data(self.src,self.dst,count=3)
        self.assertFalse(self.dst.exists());self.dst.write_bytes(b'keep-existing')
        with self.assertRaises(FileExistsError):T.convert_training_data(self.src,self.dst,count=2)
        self.assertEqual(self.dst.read_bytes(),b'keep-existing')
if __name__=='__main__':unittest.main()
