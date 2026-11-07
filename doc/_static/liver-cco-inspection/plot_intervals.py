#!/usr/bin/env python3
"""Regenerate figures using only the two small CSVs beside this script."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
rows=list(csv.DictReader((D/"interval-histogram.csv").open()))
summary=list(csv.DictReader((D/"interval-summary.csv").open()))
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","savefig.facecolor":"white"})
fig,ax=plt.subplots(figsize=(8.4,4.8),layout="constrained")
colors=["#0072B2","#D55E00","#009E73"]
names=["Original tetrahedra","Hex, local level 1","Hex, local level 2"]
for s,color,name in zip(summary,colors,names):
 r=[z for z in rows if z["case"]==s["case"]]
 edges=np.array([float(z["bin_left"]) for z in r]+[float(r[-1]["bin_right"])])
 y=np.array([float(z["fraction_inside"]) for z in r])
 label=f'{name}: {int(s["background_cells"]):,} cells; {100*float(s["lower_fraction"]):.1f}% below 0.5'
 ax.stairs(y,edges,color=color,linewidth=1.9,label=label)
ax.axvline(.5,color="#444444",linestyle="--",linewidth=1.2)
ax.text(.5,.98,"Lower threshold 0.5  ",transform=ax.get_xaxis_transform(),ha="right",va="top",fontsize=9)
ax.set_xscale("log")
ax.set_xlabel(r"Numerical interval length / maximum crossed-cell diameter, $L/H_{\max}$")
ax.set_ylabel("Fraction of interior intervals per log bin")
ax.set_ylim(bottom=0)
ax.grid(axis="x",which="major",alpha=.15)
ax.legend(loc="upper left",frameon=False,fontsize=9)
fig.suptitle("Partial A3073 geometry: numerical interval lengths",fontsize=13)
ax.set_title("Same physical network; background grids differ; 7 exterior intervals excluded per case",fontsize=9,pad=10)
for extension in ["png","svg"]:
 output=D/("interval-length-comparison."+extension)
 fig.savefig(output,dpi=240)
 if extension=="svg":
  output.write_text(chr(10).join(line.rstrip() for line in output.read_text().splitlines())+chr(10))
plt.close(fig)
