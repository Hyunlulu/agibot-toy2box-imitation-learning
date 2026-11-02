import subprocess
import sys
from pathlib import Path
import unittest


class GenerationHelpTest(unittest.TestCase):
    def test_help_succeeds_without_starting_isaac_or_requiring_output(self):
        script=Path(__file__).with_name('generate_constrained.py')
        result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--output_file',result.stdout)


if __name__=='__main__':unittest.main()
