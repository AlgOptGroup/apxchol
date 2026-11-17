"""Unranked records retain every outcome without turning missing costs into timings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('matplotlib', 'numpy')), 'plot dependencies unavailable')
class RecordsProfiles(unittest.TestCase):
    def setUp(self):
        self.docs = Path(__file__).parents[1]
        spec = importlib.util.spec_from_file_location('performance', self.docs/'performance.py')
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)
        self.headers = [p for p in (self.docs/'performance').glob('euler-*.json')
                        if json.loads(p.read_text()).get('kind') == 'records']
        self.assertEqual(len(self.headers), 3)

    def test_records_preserve_population_and_matrix_without_charts(self):
        count = 0
        with tempfile.TemporaryDirectory() as t:
            output = Path(t)
            with patch.object(self.p, 'scaling_chart', side_effect=AssertionError('record chart')):
                for header in self.headers:
                    data = json.loads(header.read_text())
                    self.p.scaling_profile(header, output)
                    table = (output/(header.stem+'-table.md')).read_text()
                    rows = [line for line in table.splitlines() if line.startswith('| ')][2:]
                    self.assertEqual(len(rows), data['provenance']['rows'])
                    self.assertIn('| Matrix | Threads |', table)
                    cells = self.p.read_cells(header.with_suffix('.csv'), data['tolerance'])
                    self.assertEqual(len(rows), len(cells))
                    cells.sort(key=lambda c: (c['matrix'], c['threads'], self.p.series_label(c)))
                    marks = {'n/a': 'n/a', 'exceeded_cap': 'cap', 'timeout': 't/o', 'missing': '—'}
                    for row, cell in zip(rows, cells):
                        fields = [field.strip() for field in row.split('|')[1:-1]]
                        self.assertEqual(fields[:3], [cell['matrix'], str(cell['threads']), self.p.series_label(cell)])
                        if cell['status'] != 'complete':
                            self.assertEqual(fields[3:6], [marks[cell['status']]] * 3)
                    count += len(rows)
            self.assertEqual(count, 115)
            self.assertEqual(list(output.glob('*.svg')), [])

    def test_records_dispatch_never_ranks_headlines(self):
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            for header in self.headers:
                (directory/header.name).write_bytes(header.read_bytes())
            with patch.object(self.p, 'scaling_profile') as profile, patch.object(self.p, 'campaign') as headline:
                self.p.campaigns(directory, directory/'output')
                self.assertEqual(profile.call_count, 3)
                headline.assert_not_called()

if __name__ == '__main__':
    unittest.main()
