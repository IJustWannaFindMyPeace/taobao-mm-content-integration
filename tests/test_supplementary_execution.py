import sys,tempfile,subprocess,unittest,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from supplementary_test_contract import single_writer,verify_dependencies,digest,validate_freeze
class ExecutionGates(unittest.TestCase):
 def test_live_lock_rejects_second_process_and_releases(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'lock';code="from supplementary_test_contract import single_writer; from pathlib import Path; import sys;\nwith single_writer(Path(sys.argv[1])): pass"
   env=__import__('os').environ.copy();env['PYTHONPATH']=str(root/'scripts')
   with single_writer(path):self.assertNotEqual(subprocess.run([sys.executable,'-c',code,str(path)],env=env,capture_output=True).returncode,0)
   self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)],env=env,capture_output=True).returncode,0)
 def test_crashed_process_releases_lock(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'lock';env=__import__('os').environ.copy();env['PYTHONPATH']=str(root/'scripts')
   code="from supplementary_test_contract import single_writer; import os,sys;\nwith single_writer(sys.argv[1]): os._exit(7)"
   self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)],env=env,capture_output=True).returncode,7)
   with single_writer(path):pass
 def test_dependency_mutation_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'src';p.mkdir();f=p/'a.py';f.write_text('before');record={'dependency_sha256':{'content/src/a.py':digest(f)}};verify_dependencies(record,Path(d));f.write_text('after')
   with self.assertRaises(ValueError):verify_dependencies(record,Path(d))
 def test_incomplete_freeze_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'freeze.json';p.write_text('{}')
   with self.assertRaises(ValueError):validate_freeze(p,p,Path(d),root)
 def test_forward_path_identical(self):
  old=(root/'scripts/replay_supplementary.py').read_text(encoding='utf-8-sig');new=(root/'scripts/evaluate_supplementary.py').read_text(encoding='utf-8-sig')
  left=old[old.index(' store=FeatureStore'):old.index(" with np.load(run/'validation_predictions.npz') as old:")]
  right=new[new.index(' store=FeatureStore'):new.index(' error=None')].replace('FeatureStore(a.store,a.split)',"FeatureStore(a.store,'validation')")
  self.assertEqual(left,right)
 def test_bootstrap_math_unchanged(self):
  old=(root/'scripts/compare_validation.py').read_text();new=(root/'scripts/compare_supplementary_test.py').read_text()
  self.assertEqual(old[old.index('    groups={}'):old.index('    result=')],new[new.index('    groups={}'):new.index('    result=')])
if __name__=='__main__':unittest.main()
