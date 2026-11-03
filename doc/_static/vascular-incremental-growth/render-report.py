#!/usr/bin/env python3
"""Validate archived native cases and export incremental-growth review assets."""
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,required=True);P.add_argument('--storage',type=Path,required=True);P.add_argument('--baseline',type=Path,required=True);P.add_argument('--output',type=Path,required=True);A=P.parse_args()
sys.path.insert(0,str(A.source/'scripts'))
from run_vascular_benchmarks import validate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'immersx-incremental-growth','axes.spines.top':False,'axes.spines.right':False})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(name,dataset):
 root=A.storage/name;m=read(root/'manifest.json')
 for filename,digest in m['hashes'].items():assert sha(root/'bin'/filename)==digest,(root,filename)
 assert sha(root/'runner.py')==m['runner_sha256'],root
 cases=[]
 for entry in m['cases']:
  path=root/entry['name']/'case.json';assert sha(path)==entry['case_sha256'],path
  r=read(path);assert r['status']=='complete' and r['returncode']==0,path
  assert sha(path.parent/'input.prm')==r['input_sha256'],path
  assert validate(path.parent/'network',r['case'])==r['physics'],path
  c=r['case'];row=dict(dataset=dataset,terminals=c['terminals'],seed=c['seed'],ranks=c['ranks'],background_cells=8**c['refinements'],generation_seconds=float(r['physics']['summary']['generation_seconds']),peak_rank_rss_kib=r['memory']['maximum_rank_peak_rss'])
  for phase,ops in r['work'].items():
   for k,v in ops.items():row[phase+'_'+k]=v
  for k in r['physics']['growth_rank_totals'][0]:row[k]=max(x[k] for x in r['physics']['growth_rank_totals'])
  phase_rows=r['phase_mpi_by_rank'];row['mpi_calls']=sum(int(x['calls']) for rr in phase_rows for x in rr);row['mpi_posted_sends']=sum(int(x['posted_sends']) for rr in phase_rows for x in rr);row['mpi_logical_bytes']=sum(int(x['posted_send_bytes'])+int(x['collective_input_bytes']) for rr in phase_rows for x in rr);row['mpi_reductions']=sum(int(x['reductions']) for rr in phase_rows for x in rr)
  # Offline physical graph depth is independent of engine dependency counters.
  from render_vascular_figures import keyed,tables
  nodes=keyed(tables(path.parent/'network','nodes'));vessels=keyed(tables(path.parent/'network','vessels'));children={i:[] for i in nodes};parent=set()
  for v in vessels.values():children[int(v['node0'])].append(int(v['node1']));parent.add(int(v['node1']))
  depth={i:0 for i in set(nodes)-parent};order=list(depth)
  for i in order:
   for child in children[i]:assert child not in depth;depth[child]=depth[i]+1;order.append(child)
  assert len(depth)==len(nodes)
  leaves=[depth[i] for i in depth if not children[i]];row['maximum_leaf_depth']=max(leaves);row['mean_leaf_depth']=statistics.mean(leaves)
  cases.append(row)
 return cases,dict(directory=name,manifest_sha256=sha(root/'manifest.json'),source_commit=m['commit'],binary_sha256=m['hashes'],cases=len(cases))
records=[];provenance=[]
for name,label in [('incremental-growth-production-pilot','pilot'),('incremental-growth-production-strong','serial'),('incremental-growth-production-background','background'),('incremental-growth-production-mpi','mpi-small'),('incremental-growth-production-mpi-1000','mpi-large')]:
 rows,identity=load(name,label);records+=rows;provenance.append(identity)
base=read(A.baseline);old=[]
for group in base['groups'].values():
 assert sha(Path(group['manifest']))==group['manifest_sha256']
 manifest=read(Path(group['manifest']));directory=Path(group['manifest']).parent
 for case in manifest['identity']['cases']:
  if case['ranks']!=1:continue
  report=directory/case['name']/'case.json';r=read(report);assert r['status']=='complete' and r['returncode']==0
  result=validate(directory/case['name']/'network',case)
  old.append(dict(dataset='historical',terminals=case['terminals'],seed=case['seed'],ranks=1,background_cells=512,generation_seconds=float(result['summary']['generation_seconds']),peak_rank_rss_kib=max(int(x['peak_rss_kb']) for x in result['ranks'])))
records+=old
A.output.mkdir(parents=True,exist_ok=False)
columns=['dataset','terminals','seed','ranks','background_cells','generation_seconds','peak_rank_rss_kib']+sorted(set().union(*(r.keys() for r in records))-{'dataset','terminals','seed','ranks','background_cells','generation_seconds','peak_rank_rss_kib'})
with (A.output/'measurements.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows(records)
def series(rows,key,xkey='terminals'):
 result=[]
 for n in sorted({r[xkey] for r in rows}):
  v=[r[key] for r in rows if r[xkey]==n];result.append((n,statistics.median(v),min(v),max(v)))
 return result
def curve(ax,rows,key,label,color,xkey='terminals',marker='o'):
 values=series(rows,key,xkey);x,y,lo,hi=zip(*values);ax.plot(x,y,marker=marker,color=color,label=label);ax.fill_between(x,lo,hi,color=color,alpha=.13);return values
def export(fig,name):
 fig.tight_layout();fig.savefig(A.output/(name+'.svg'),metadata={'Date':None});svg=A.output/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');fig.savefig(A.output/(name+'.png'),dpi=160,metadata={'Software':'ImmersX native measurement report'});plt.close(fig)
new=[r for r in records if r['dataset']=='serial'];hist=[r for r in records if r['dataset']=='historical']
fig,axes=plt.subplots(1,2,figsize=(11,4.1));ax=axes[0]
curve(ax,new,'generation_seconds','Incremental: 4 sizes, 3 seeds','#1768ac');curve(ax,hist,'generation_seconds','Historical: 2 sizes, 3 seeds','#bd452d',marker='s');ax.set(xscale='log',yscale='log',xlabel='Terminal count N',ylabel='Native generation time [s]',title='Same SI point-demand protocol');ax.legend(fontsize=8);ax.grid(alpha=.18)
ax=axes[1]
for key,label,color in [('optimization_seconds','Candidate optimization','#1768ac'),('search_seconds','Candidate/path/obstacle search','#7954a1'),('commit_seconds','Incremental commit','#398339'),('finalization_seconds','Final reconstruction','#bd452d')]:curve(ax,new,key,label,color)
ax.set(xscale='log',yscale='log',xlabel='Terminal count N',ylabel='Recorded phase time [s]',title='Phases shown separately, not stacked');ax.legend(fontsize=8);ax.grid(alpha=.18);export(fig,'timing')
fig,axes=plt.subplots(1,2,figsize=(11,4.1));ax=axes[0]
for key,label,color in [('ordinary_vessel_midpoint_localizations','Midpoint locations','#1768ac'),('ordinary_certified_vessels','Certified rows','#398339'),('ordinary_physiology_branch_visits','Physiology dependency visits','#7954a1')]:curve(ax,new,key,label,color)
ax.set(xscale='log',yscale='log',xlabel='Terminal count N',ylabel='Summed ordinary operations',title='Ordinary insertion work');ax.legend(fontsize=8);ax.grid(alpha=.18)
ax=axes[1]
for key,label,color in [('ordinary_candidate_path_visits','Candidate path visits / N','#1768ac'),('ordinary_pair_checks','Geometric pair checks / N','#398339'),('ordinary_physiology_branch_visits','Physiology visits / N','#7954a1')]:
 rows=[dict(r,per_terminal=r[key]/r['terminals']) for r in new];curve(ax,rows,'per_terminal',label,color)
ax.set(xscale='log',yscale='log',xlabel='Terminal count N',ylabel='Ordinary operations / terminal',title='Remaining work per terminal');ax.legend(fontsize=8);ax.grid(alpha=.18);export(fig,'work')
fig,axes=plt.subplots(1,2,figsize=(11,4.1));ax=axes[0]
mesh=[r for r in records if r['dataset']=='background'];curve(ax,mesh,'initialization_background_cell_visits','Initialization cell visits','#1768ac',xkey='background_cells');curve(ax,mesh,'ordinary_vessel_midpoint_localizations','Ordinary midpoint locations','#398339',xkey='background_cells');ax.text(.04,.1,'Ordinary background-cell visits = 0\nOrdinary full materialization = 0',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85));ax.set(xscale='log',yscale='log',xlabel='Background cells M (N=128)',ylabel='Operations',title='Fixed physical network across backgrounds');ax.legend(fontsize=8);ax.grid(alpha=.18)
ax=axes[1]
for rows,label,color in [(new,'Incremental','#1768ac'),(hist,'Historical','#bd452d')]:curve(ax,[dict(r,memory_mib=r['peak_rank_rss_kib']/1024) for r in rows],'memory_mib',label,color)
ax.set(xscale='log',xlabel='Terminal count N',ylabel='Maximum rank peak RSS [MiB]',title='Native process high-water mark');ax.legend(fontsize=8);ax.grid(alpha=.18);export(fig,'background-memory')
summary={}
for n in sorted({r['terminals'] for r in new}):
 rows=[r for r in new if r['terminals']==n];summary[n]={k:dict(median=statistics.median(r[k] for r in rows),minimum=min(r[k] for r in rows),maximum=max(r[k] for r in rows)) for k in ['generation_seconds','peak_rank_rss_kib','ordinary_physical_copy_attempts','ordinary_vessel_midpoint_localizations','ordinary_certified_vessels','ordinary_physiology_branch_visits','ordinary_candidate_path_visits','ordinary_pair_checks','mean_leaf_depth','maximum_leaf_depth']}
save(A.output/'summary.json',summary)
manifest=dict(purpose='Measured native SI point-demand execution; integrated-demand timings are a separate experiment',report_script_sha256=sha(Path(__file__)),matplotlib_version=matplotlib.__version__,provenance=provenance,baseline_verified_sha256=sha(A.baseline),files={p.name:sha(p) for p in A.output.iterdir() if p.is_file()},notes=['Bands span all three seeds; no fit','Phase curves overlap only as documented; do not add nested commit subphases','Maximum per-rank peak RSS is not summed node memory','Historical cases contain only the two recorded terminal counts','Dataset source is frozen before the later include-only CI compatibility fix'])
save(A.output/'manifest.json',manifest)
print(json.dumps(summary,indent=2))
