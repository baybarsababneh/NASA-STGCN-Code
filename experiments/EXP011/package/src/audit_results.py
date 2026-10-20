"""Read-only post-run byte/count verification; scientific review remains necessary."""
import argparse,json,itertools
from pathlib import Path
from pilot import filehash,canonical,digest,MECH,ARMS,SEEDS
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
m=json.loads((root/'OUTPUT_MANIFEST.json').read_text())
for f,h in m.items():assert filehash(root/f)==h,f
fits=list((root/'fits').glob('*.json'));assert len(fits)==440
freeze=json.loads((root/'RUN_FREEZE.json').read_text());rh=digest(canonical(freeze));expected=set()
for mission,roster in MECH.items():
 for size in (4,5,6):
  for channels in itertools.combinations(roster,size):
   for arm in ARMS:
    for seed in ((17,) if arm=='persistence' else SEEDS):
     expected.add(digest(canonical(dict(runhash=rh,mission=mission,channels=tuple(sorted(channels)),seed=seed,arm=arm)))[:24])
identities=set()
for f in fits:
 r=json.loads(f.read_text());identity=r['identity'];jid=digest(canonical(identity))[:24];assert f.stem==jid and jid not in identities;identities.add(jid)
 assert digest(canonical(r['payload']))==r['payload_sha256']
assert identities==expected,'Unexpected/missing scientific fit identities'
c=json.loads((root/'STUDY_CLOSE.json').read_text());assert c['detector_fits']==440 and c['forecasting_fits']==0 and not c['NASA_TEST_accessed'] and not c['final_calibration_accessed']
print('PASS: output manifest and 440 unique committed detector fits verified; no deployment detector selected.')
