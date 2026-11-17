"""Scaling-only profiles keep thread counts, destinations and flagged observations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

class ScalingProfiles(unittest.TestCase):
    def setUp(self):
        self.docs=Path(__file__).parents[1]
        spec=importlib.util.spec_from_file_location('performance',self.docs/'performance.py')
        self.p=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.p)
        self.headers=sorted((self.docs/'performance').glob('euler-2117494*-descriptive-scaling.json'))
        self.assertEqual(len(self.headers),2)
    def test_separate_outputs_and_all_thread_rows(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t);old=output/'euler-scaling.svg';old.write_text('historical sentinel')
            with patch.object(self.p,'Campaign',side_effect=AssertionError('headline path used')):
                for header in self.headers:self.p.scaling_profile(header,output)
            self.assertEqual(old.read_text(),'historical sentinel')
            for header in self.headers:
                table=(output/(header.stem+'-table.md')).read_text()
                self.assertEqual(sum(line.startswith('| ') for line in table.splitlines()),42)
                self.assertTrue((output/(header.stem+'-scaling.svg')).exists())
                for threads in [1,2,4,8,16,32,64,128]:self.assertEqual(table.count(f'| {threads} |'),5)
            ipm=(output/(self.headers[1].stem+'-table.md')).read_text()
            self.assertIn('ParAC (Physics, portable, shared RHS)',ipm)
            self.assertIn('23.260550',ipm)
    def test_flag_is_retained_but_breaks_the_trend(self):
        captured=[]
        with tempfile.TemporaryDirectory() as t, patch.object(self.p,'save',side_effect=lambda fig,path:captured.append(fig)):
            self.p.scaling_profile(self.headers[1],Path(t))
        fig=captured[0];lines=fig.axes[0].get_lines()
        trends=[line for line in lines if len(line.get_xdata())==8]
        self.assertEqual(len(trends),5)
        self.assertEqual(sum(np.isnan(line.get_ydata()).sum() for line in trends),1)
        flagged=[line for line in lines if line.get_marker()=='D']
        self.assertEqual(len(flagged),1)
        self.assertEqual(list(flagged[0].get_xdata()),[32])
        self.assertGreater(flagged[0].get_ydata()[0],26)
        self.p.plt.close(fig)
    def test_dispatch_uses_scaling_path(self):
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            for header in self.headers:(directory/header.name).write_bytes(header.read_bytes())
            with patch.object(self.p,'scaling_profile') as scaling, patch.object(self.p,'campaign') as headline:
                self.p.campaigns(directory,directory/'output')
                self.assertEqual(scaling.call_count,2);headline.assert_not_called()
if __name__=='__main__':unittest.main()
