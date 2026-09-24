"""Immutable artifact checks and OS-backed single-writer lock for test execution."""
import contextlib,hashlib,json,os
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def verify_dependencies(frozen,contentroot):
 roots={'content':contentroot,'muse':contentroot.parent/'taobao-mm-muse-repro'}
 for key,value in frozen['dependency_sha256'].items():
  project,relative=key.split('/',1)
  if digest(roots[project]/relative)!=value:raise ValueError('Frozen dependency changed: '+key)

def validate_freeze(path,inventory,store,contentroot):
 f=json.loads(Path(path).read_text())
 if f.get('scope')!='frozen_supplementary_test_v1' or not f.get('all_validation_replays_exact'):raise ValueError('Test freeze gate absent')
 if f['candidate_inventory_sha256']!=digest(inventory):raise ValueError('Inventory changed')
 if f['store_manifest_sha256']!=digest(Path(store)/'manifest.json'):raise ValueError('Store changed')
 verify_dependencies(f,contentroot)
 for name,value in f['test_arrays_sha256'].items():
  if digest(Path(store)/name)!=value:raise ValueError('Test data changed: '+name)
 return f

@contextlib.contextmanager
def single_writer(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 stream=path.open('a+b')
 if path.stat().st_size==0:stream.write(b'0');stream.flush()
 stream.seek(0)
 try:
  if os.name=='nt':
   import msvcrt
   msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl
   fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except (OSError,BlockingIOError):
  stream.close();raise RuntimeError('Another process owns this output directory')
 try:yield
 finally:
  stream.seek(0)
  if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
  else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
  stream.close()
