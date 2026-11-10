# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
import csv,json,hashlib,re,argparse
p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
O=a.output.resolve();O.mkdir(exist_ok=True);allrows=[];statuses=[];hashes={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for C in a.cases:
 C=C.resolve();m=json.loads((C/'run.json').read_text());text=(C/'parameters.prm').read_text();hashes[str(C/'run.json')]=sha(C/'run.json');hashes[str(C/'parameters.prm')]=sha(C/'parameters.prm')
 status=m.get('status');exit=m.get('exit_code');statuses.append({'case':C.name,'native_status':status,'exit_code':exit,'plotted':status=='FINISHED' and exit==0})
 if status!='FINISHED' or exit!=0:continue
 D=C/'generated';tables={}
 for prefix in ['terminals','triplets']:
  rows=[];paths=sorted(D.glob(prefix+'-*.csv'));assert len(paths)==m['mpi_ranks']
  for f in paths:rows+=list(csv.DictReader(f.open()));hashes[str(f)]=sha(f)
  tables[prefix]=rows
 T={int(x['id']):x for x in tables['terminals']};beds=tables['triplets'];ds=[float(T[int(b['arterial'])]['demand']) for b in beds];S=sum(ds);N=len(beds);assert len(T)==3*N
 get=lambda key:re.search(r'set '+re.escape(key)+r'\s*=\s*([^\n]+)',text).group(1).strip()
 total=float(get('Total new-tissue reference flow'));fp=float(get('Flow fraction'));Q=[total*(1-fp),total*fp,total]
 scaled='set Scale compartment resistances by demand = true' in text
 label=str(N)+' territories; '+('demand-scaled arms' if scaled else 'uniform arms ×'+str(m.get('resistance_scale_sensitivity',1)))
 if scaled and m.get('resistance_scale_sensitivity_by_arm',{}).get('portal',1)!=1:label+='; PV arm ×'+str(m['resistance_scale_sensitivity_by_arm']['portal'])
 for b,d in zip(beds,ds):
  qs=[float(b[k+'_flow']) for k in ['arterial','portal','venous']];assert min(qs)>0
  r={'case_label':label,'case_id':C.name,'territory':int(b['territory']),'arterial_terminal':int(b['arterial']),'portal_terminal':int(b['portal']),'venous_terminal':int(b['venous']),'territories':N,'demand_weight':d/S,'global_portal_fraction':fp,'local_portal_fraction':qs[1]/qs[2]}
  for role,q,qtot in zip(['arterial','portal','venous'],qs,Q):r[role+'_actual_over_demand_design']=q/(qtot*d/S);r[role+'_flow_m3_s']=q;r[role+'_arm_resistance_Pa_s_m3']=float(b[role+'_resistance'])
  allrows.append(r)
assert allrows,'No finished valid results to plot'
with (O/'perfusion-distribution.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (O/'case-status.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(statuses[0]));w.writeheader();w.writerows(statuses)
(O/'provenance.json').write_text(json.dumps({'source_sha256':hashes,'scope':'Native rigid outputs; local flows arise from resistance graph. Failed/pending cases have status only and no invented numeric data. No coupled or clinical validation claim.'},indent=2)+'\n')
print('Collected',len(allrows),'rows;',statuses)
