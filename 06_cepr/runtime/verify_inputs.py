"""Verify exact image and SAR/registration inputs, without opening target labels."""
import argparse,csv,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def verify_aux(data_root,scene):
 root=Path(data_root).resolve();aux=root/'auxiliary';contract=json.loads((B/'auxiliary_contract.json').read_text())
 with (aux/'availability.csv').open(encoding='utf-8-sig',newline='') as f:rows={r['scene_stem']:r for r in csv.DictReader(f)}
 expected=contract['scenes'][scene];assert rows[scene]==expected['availability_row'],('Auxiliary scene metadata changed',scene)
 for row in expected['files']:
  path=(aux/row['path']).resolve();assert path.is_relative_to(aux),row['path']
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],('Auxiliary file changed',scene,str(path))
 return len(expected['files'])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);a=p.parse_args();root=Path(a.data_root)
 images=json.loads((B/'official30_images.json').read_text())['scenes'];count=0
 for row in images:
  assert sha(root/'images'/(row['id']+'.tif'))==row['image_sha256'],('MS image changed',row['id'])
  count+=verify_aux(root,row['id'])
 print(json.dumps(dict(status='passed',scenes=len(images),auxiliary_file_checks=count,labels_read=False)))
if __name__=='__main__':main()
