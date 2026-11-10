"""Win-SUDS contract tests; external Windows binaries are not required."""
import importlib.util
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from flovopy.core import winsuds as w
except ImportError:
    spec=importlib.util.spec_from_file_location('winsuds',Path(__file__).with_name('winsuds.py'))
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class WinSUDSTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.source=self.root/'original with spaces.DMX';self.source.write_bytes(b'original')
        self.tools=self.root/'tool folder';self.tools.mkdir()
        for name in w.TOOLS:(self.tools/(name+'.exe')).write_bytes(b'not a real executable')
        self.options=dict(cwd=self.root,tool_dir=self.tools)
        self.calls=[]
    def simulate(self,cmd,**kwargs):
        self.calls.append((cmd,kwargs));stage=Path(kwargs['cwd']);tool=Path(cmd[0]).stem
        self.assertNotEqual(stage,self.root);self.assertEqual(Path.cwd(),self.start_dir)
        self.assertEqual((stage/('input'+self.source.suffix)).read_bytes(),b'original')
        self.assertFalse(kwargs['shell']);self.assertEqual(kwargs['errors'],'replace')
        if tool=='irig':
            (stage/('input'+self.source.suffix)).write_bytes(b'corrected')
            (stage/'IRIG.LOG').write_text('decode status')
            if '/D' in cmd:(stage/'IRIG.ASC').write_text('diagnostics')
        elif tool in ['demux','sud2msed'] or (tool=='sud2asc' and cmd[-1].startswith('output')):
            (stage/cmd[-1]).write_bytes(b'converted')
        elif tool in ['sud2sac','sud2gse','sud2mat']:
            endings={'sud2sac':['.S00','.S01'],'sud2gse':['.GSE'],'sud2mat':['.MAT']}[tool]
            for ext in endings:(stage/('input'+ext)).write_bytes(b'converted')
        return subprocess.CompletedProcess(cmd,0,'ASCII output','')
    def running(self):
        self.start_dir=Path.cwd()
        return patch.object(w.subprocess,'run',side_effect=self.simulate)
    def test_all_conversions_and_persistent_results(self):
        functions=[lambda:w.demux(self.source,self.root/'demuxed.DMX',bits=12,**self.options),
            lambda:w.sud2msed(self.source,'converted.mseed',**self.options),
            lambda:w.sud2sac(self.source,little_endian=True,**self.options),
            lambda:w.sud2gse(self.source,**self.options),lambda:w.sud2mat(self.source,**self.options),
            lambda:w.sud2asc(self.source,'converted.txt',no_data=True,quiet=True,date_format='U',**self.options)]
        with self.running():
            for fn in functions:
                result=fn();self.assertIsInstance(result,subprocess.CompletedProcess)
                self.assertTrue(result.output_files)
                self.assertTrue(all(p.is_file() for p in result.output_files))
                self.assertFalse(Path(self.calls[-1][1]['cwd']).exists())
        self.assertEqual(self.source.read_bytes(),b'original')
        self.assertTrue((self.root/'original with spaces.S00').is_file())
        self.assertIn('/12',self.calls[0][0]);self.assertIn('-l',self.calls[2][0])
        self.assertEqual(self.calls[-1][0][1:4],['/N','/Q','/DU'])
    def test_ascii_stdout_no_file(self):
        with self.running():r=w.sud2asc(self.source,**self.options)
        self.assertEqual(r.stdout,'ASCII output');self.assertFalse(r.output_files)
    def test_irig_preserves_input_by_default(self):
        with self.running():r=w.irig(self.source,station='IRIG:z',diagnostics=True,**self.options)
        self.assertEqual(self.source.read_bytes(),b'original')
        self.assertEqual(r.output_files[0].read_bytes(),b'corrected')
        self.assertEqual(len(r.output_files),3);self.assertIn('IRIG:z',r.args)
    def test_irig_explicit_inplace(self):
        with self.running():r=w.irig(self.source,inplace=True,**self.options)
        self.assertEqual(self.source.read_bytes(),b'corrected');self.assertEqual(r.output_files[0],self.source)
    def test_irig_failed_run_never_modifies_source(self):
        def bad(cmd,**kwargs):
            self.simulate(cmd,**kwargs);return subprocess.CompletedProcess(cmd,1,'bad','failed decode')
        self.start_dir=Path.cwd()
        with patch.object(w.subprocess,'run',side_effect=bad),self.assertRaises(w.WinSUDSError):w.irig(self.source,inplace=True,**self.options)
        self.assertEqual(self.source.read_bytes(),b'original');self.assertFalse((self.root/'original with spaces.irig.log').exists())
    def test_irig_diagnostics_missing(self):
        def no_ascii(cmd,**kwargs):
            result=self.simulate(cmd,**kwargs);(Path(kwargs['cwd'])/'IRIG.ASC').unlink();return result
        self.start_dir=Path.cwd()
        with patch.object(w.subprocess,'run',side_effect=no_ascii),self.assertRaises(w.WinSUDSError):w.irig(self.source,diagnostics=True,**self.options)
        self.assertEqual(self.source.read_bytes(),b'original')
    def test_irig_inplace_does_not_implicitly_overwrite_logs(self):
        log=self.root/'original with spaces.irig.log';log.write_text('old log')
        with self.running(),self.assertRaises(FileExistsError):w.irig(self.source,inplace=True,**self.options)
        self.assertEqual(log.read_text(),'old log');self.assertEqual(self.source.read_bytes(),b'original')
    def test_existing_output_not_used_as_success(self):
        dst=self.root/'already.mseed';dst.write_bytes(b'old')
        with patch.object(w.subprocess,'run') as run,self.assertRaises(FileExistsError):w.sud2msed(self.source,dst,**self.options)
        run.assert_not_called()
        with patch.object(w.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')),self.assertRaises(w.WinSUDSError):w.sud2msed(self.source,dst,overwrite=True,**self.options)
        self.assertEqual(dst.read_bytes(),b'old')
    def test_overwrite_only_after_valid_output(self):
        dst=self.root/'already.mseed';dst.write_bytes(b'old')
        with self.running():w.sud2msed(self.source,dst,overwrite=True,**self.options)
        self.assertEqual(dst.read_bytes(),b'converted')
    def test_implicit_group_collision_publishes_nothing(self):
        dst=self.root/'original with spaces.S01';dst.write_bytes(b'old')
        with self.running(),self.assertRaises(FileExistsError):w.sud2sac(self.source,**self.options)
        self.assertFalse((self.root/'original with spaces.S00').exists());self.assertEqual(dst.read_bytes(),b'old')
    def test_empty_output_rejected(self):
        def empty(cmd,**kwargs):
            (Path(kwargs['cwd'])/cmd[-1]).touch();return subprocess.CompletedProcess(cmd,0,'empty','')
        with patch.object(w.subprocess,'run',side_effect=empty),self.assertRaises(w.WinSUDSError):w.sud2msed(self.source,'empty.mseed',**self.options)
        self.assertFalse((self.root/'empty.mseed').exists())
    def test_missing_implicit_output(self):
        with patch.object(w.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')),self.assertRaises(w.WinSUDSError):w.sud2mat(self.source,**self.options)
    def test_missing_executable(self):
        (self.tools/'demux.exe').unlink()
        with self.assertRaises(w.WinSUDSError):w.demux(self.source,'missing-exe.DMX',**self.options)
        self.assertIsNone(w.find_tools(self.tools)['demux'])
    def test_nonzero_diagnostics_preserved(self):
        with patch.object(w.subprocess,'run',return_value=subprocess.CompletedProcess(['x'],7,'out','err')):
            with self.assertRaises(w.WinSUDSError) as caught:w.sud2asc(self.source,**self.options)
        self.assertEqual(caught.exception.returncode,7);self.assertEqual(caught.exception.stderr,'err')
    def test_timeout_diagnostics_preserved(self):
        with patch.object(w.subprocess,'run',side_effect=subprocess.TimeoutExpired(['x'],2,output=b'partial',stderr=b'bad')):
            with self.assertRaises(w.WinSUDSError) as caught:w.sud2asc(self.source,timeout=2,**self.options)
        self.assertEqual(caught.exception.stdout,'partial')
    def test_launch_error_wrapped(self):
        with patch.object(w.subprocess,'run',side_effect=OSError('Exec format error')),self.assertRaises(w.WinSUDSError):w.sud2asc(self.source,**self.options)
    def test_argument_and_input_validation(self):
        for value in [True,8,16.0]:
            with self.assertRaises(ValueError):w.demux(self.source,bits=value,**self.options)
        with self.assertRaises(ValueError):w.sud2asc(self.source,date_format='bad',**self.options)
        with self.assertRaises(ValueError):w.irig(self.source,station='/D',**self.options)
        with self.assertRaises(ValueError):w.demux('*.DMX',**self.options)
        with self.assertRaises(FileNotFoundError):w.demux('missing.DMX',**self.options)
        with self.assertRaises(ValueError):w.sud2msed(self.source,self.source,overwrite=True,**self.options)
        with self.assertRaises(ValueError):w.irig(self.source,output_file='copy',inplace=True,**self.options)
        for value in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):w.sud2asc(self.source,timeout=value,**self.options)
    @unittest.skipIf(os.name == 'nt', 'POSIX fixture; real Win-SUDS integration is separate')
    def test_real_subprocess_smoke(self):
        exe=self.tools/'sud2msed.exe'
        exe.write_text('#!'+sys.executable+'\nfrom pathlib import Path\nimport sys\nPath(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes()+b" converted")\nprint("done")\n')
        exe.chmod(0o755)
        before=Path.cwd()
        r=w.sud2msed(self.source,'real process.mseed',timeout=5,**self.options)
        self.assertEqual(r.output_files[0].read_bytes(),b'original converted')
        self.assertEqual(r.stdout.strip(),'done');self.assertEqual(Path.cwd(),before)

    def test_relative_paths_follow_cwd(self):
        with self.running():r=w.sud2msed(self.source.name,'relative.mseed',**self.options)
        self.assertEqual(r.output_files,(self.root/'relative.mseed',))

if __name__=='__main__':unittest.main()
