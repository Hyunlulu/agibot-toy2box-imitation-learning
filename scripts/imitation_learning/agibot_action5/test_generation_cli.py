from pathlib import Path
import tempfile
import unittest
try:
    from generation_cli import generation_arguments
except ImportError:
    generation_arguments=None


class GenerationArgumentsTest(unittest.TestCase):
    def call(self,args):
        self.assertTrue(callable(generation_arguments),'safe generation argument handling is not implemented')
        return generation_arguments(args)

    def test_effective_output_aliases_never_overwrite_actual_hdf5_destination(self):
        with tempfile.TemporaryDirectory() as d:
            existing=Path(d)/'existing.hdf5';existing.write_bytes(b'keep this dataset')
            for value in ['existing','existing.txt','existing.hdf5']:
                with self.subTest(value=value),self.assertRaises(FileExistsError):
                    self.call(['--output_file',str(Path(d)/value)])
            with self.assertRaises(FileExistsError):
                self.call(['--output_file',str(Path(d)/'fresh.hdf5'),'--output_file',str(existing)])
            self.assertEqual(existing.read_bytes(),b'keep this dataset')

    def test_omitted_environment_count_uses_actual_production_sixteen(self):
        with tempfile.TemporaryDirectory() as d:
            args,output=self.call(['--output_file',str(Path(d)/'new.txt'),'--headless'])
            self.assertEqual(output,Path(d)/'new.hdf5')
            self.assertEqual(args[args.index('--num_envs')+1],'16')
            self.assertEqual(args[args.index('--generation_num_trials')+1],'1000')

    def test_conflicting_effective_environment_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for options in [['--num_envs','1'],['--num_envs','16','--num_envs','1']]:
                with self.subTest(options=options),self.assertRaises(ValueError):
                    self.call(['--output_file',str(Path(d)/'new.hdf5'),*options])

    def test_existing_failed_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            failed=Path(d)/'new_failed.hdf5';failed.write_bytes(b'keep failures')
            with self.assertRaises(FileExistsError):self.call(['--output_file',str(Path(d)/'new')])
            self.assertEqual(failed.read_bytes(),b'keep failures')


if __name__=='__main__':unittest.main()
