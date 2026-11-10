# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=Path(__file__).resolve().parent
rows=list(csv.DictReader((O/'perfusion-distribution.csv').open()))
cases=list(dict.fromkeys(a['case_label'] for a in rows))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors={'arterial':'#c83b3b','portal':'#31834c','venous':'#356cb0'}
markers={'arterial':'o','portal':'^','venous':'s'}
labels={'arterial':'HA','portal':'PV','venous':'HV'}
fig,axes=plt.subplots(1,len(cases),figsize=(4.5*len(cases),3.8),squeeze=False,constrained_layout=True)
for ax,label in zip(axes[0],cases):
 rr=[a for a in rows if a['case_label']==label];w=np.array([float(a['demand_weight']) for a in rr])
 for role in colors:
  y=w*np.array([float(a[role+'_actual_over_demand_design']) for a in rr])
  ax.scatter(w,y,s=19,alpha=.7,color=colors[role],marker=markers[role],label=labels[role],linewidths=0)
 allv=[float(a['demand_weight']) for a in rr]+[float(a['demand_weight'])*float(a[r+'_actual_over_demand_design']) for a in rr for r in colors]
 lo=min(allv)*.75;hi=max(allv)*1.3
 ax.plot([lo,hi],[lo,hi],color='.35',ls='--',lw=.9)
 ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Territory demand / total demand',ylabel='Actual role flow / total role flow',title=label.replace('; ',';\n'))
 ax.set_aspect('equal',adjustable='box');ax.grid(which='major',alpha=.2);ax.legend(frameon=False,ncol=3,loc='upper left')
for ext in ['png','svg']:fig.savefig(O/('demand-versus-perfusion.'+ext),dpi=220)
fig2,axes=plt.subplots(1,len(cases),figsize=(4.5*len(cases),3.3),squeeze=False,constrained_layout=True)
bins=np.linspace(0,1,21)
for ax,label in zip(axes[0],cases):
 rr=[a for a in rows if a['case_label']==label]
 ax.hist([float(a['local_portal_fraction']) for a in rr],bins=bins,color=colors['portal'],edgecolor='white',linewidth=.6)
 ax.axvline(float(rr[0]['global_portal_fraction']),ls='--',color='.25',lw=1,label='Global portal fraction')
 ax.set(xlim=(0,1),xlabel=r'Local portal fraction $q_{PV}/q_{HV}$',ylabel='Number of territories',title=label.replace('; ',';\n'))
 ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
for ext in ['png','svg']:fig2.savefig(O/('portal-fraction-distribution.'+ext),dpi=220)
print('Rendered',len(rows),'territories in',len(cases),'valid result(s)')
