"""Draws Figures 1-3 of the manuscript (pipeline, tau sensitivity, robustness).
Run:  python make_figures.py   (writes fig_*.png next to this file)

The plotted numbers are copied verbatim from the result files of the repository:
  Figure 2 (tau sensitivity)  <- results/tau_sweep.csv
  Figure 3 (robustness)       <- results/robust_occlusion_table.csv, results/robust_temporal_table.csv
Figure 1 is a schematic and uses no data.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]
INK="#2b2b2b"; MUTED="#6b6b6b"; GRID="#e4e4e4"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":MUTED,"axes.labelcolor":INK,
 "xtick.color":MUTED,"ytick.color":MUTED,"axes.spines.top":False,"axes.spines.right":False})

# ---------- Fig: pipeline
fig,ax=plt.subplots(figsize=(9.5,3.2)); ax.set_xlim(0,100); ax.set_ylim(0,40); ax.axis("off")
boxes=[(1,22,15,14,"1. Appearance\nbackbone","ResNet-50 + BNNeck\nCE + triplet\n(S04 or VeRi-776)"),
       (20,22,15,14,"2. Single-camera\ntracklets","GT tracklets (B) or\nKalman + Hungarian\non detections (C)"),
       (39,22,17,14,"3. Tracklet scene\ngraph","cross-camera pairs in\nlearned time window;\nk-NN + cosine > τ"),
       (60,22,17,14,"4. Association\nnetwork","temporal encoder +\ntransition-aware GAT;\nedge probability p_ij"),
       (81,22,18,14,"5. Constrained\nclustering","average linkage ≥ θ;\ncannot-link same\ncamera & time")]
for x,y,w,h,t,s in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc="#eef4fc",ec=C[0],lw=1.2))
    ax.text(x+w/2,y+h-3.2,t,ha="center",va="center",fontsize=7.4,fontweight="bold",color=INK)
    ax.text(x+w/2,y+4.6,s,ha="center",va="center",fontsize=6.0,color=MUTED)
for (x1,_,w1,_,_,_),(x2,_,_,_,_,_) in zip(boxes,boxes[1:]):
    ax.add_patch(FancyArrowPatch((x1+w1+0.6,29),(x2-0.6,29),arrowstyle="-|>",mutation_scale=10,color=INK,lw=1.2))
ax.add_patch(FancyBboxPatch((20,3),79,11,boxstyle="round,pad=0.4,rounding_size=1.2",fc="#fff6e6",ec=C[3],lw=1.2))
ax.text(59.5,11.3,"Evaluation (official-style MTMC protocol, all cameras in one accumulator)",ha="center",fontsize=7.2,fontweight="bold",color=INK)
ax.text(59.5,6.4,"leak-free scene split · 2-fold cross-validation S02 ↔ S05 for epoch, τ, θ\nIDF1 / IDP / IDR / MOTA · 3 seeds + paired bootstrap",ha="center",va="center",fontsize=6.2,color=MUTED)
ax.add_patch(FancyArrowPatch((90,21.4),(90,14.8),arrowstyle="-|>",mutation_scale=10,color=INK,lw=1.2))
fig.savefig("fig_pipeline.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Fig: tau sensitivity (S04 study, final model)
tau=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
e2=[5507,5272,4915,4378,3698,2914,2191,1578,1121,773,508,302,176,95,41,13,4]
e5=[24494,24398,24225,23867,23269,21818,18954,14686,10507,6794,3956,2122,1132,569,266,102,27]
i2=[.6791,.6791,.6791,.6791,.6791,.6741,.6805,.6831,.6892,.6893,.6893,.6844,.6747,.6958,.6914,.6851,.6829]
i5=[.4559,.4549,.4559,.4554,.4551,.4598,.4572,.4542,.4571,.4533,.4535,.4484,.4361,.4273,.4101,.3983,.3873]
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.7))
for ax in (a,b): ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
a.plot(tau,e2,"-o",color=C[0],lw=2,ms=4,label="S02"); a.plot(tau,e5,"-s",color=C[1],lw=2,ms=4,label="S05")
a.set_yscale("log"); a.set_xlabel("graph threshold τ"); a.set_ylabel("edges kept (log scale)"); a.legend(frameon=False)
a.set_title("(a) graph size",loc="left",fontsize=9,color=INK)
b.plot(tau,i2,"-o",color=C[0],lw=2,ms=4,label="S02"); b.plot(tau,i5,"-s",color=C[1],lw=2,ms=4,label="S05")
b.set_xlabel("graph threshold τ"); b.set_ylabel("IDF1"); b.set_ylim(0.35,0.75); b.legend(frameon=False)
b.set_title("(b) association quality",loc="left",fontsize=9,color=INK)
fig.tight_layout(); fig.savefig("fig_tau.png",dpi=300); plt.close(fig)

# ---------- Fig: robustness (S04 study)
V=[("Appearance only (cosine)",C[0],"o"),("GAT + attention pooling",C[2],"^"),("Transition-aware GAT + LSTM",C[3],"D"),("Transition-aware GAT + attention (final)",C[1],"s")]
occ={"Appearance only (cosine)":([.4650,.4602,.4040],[0,0,0]),"GAT + attention pooling":([.4801,.4717,.4005],[.006,.0065,.007]),
     "Transition-aware GAT + LSTM":([.4558,.4568,.3889],[.0061,.0008,.0048]),"Transition-aware GAT + attention (final)":([.4835,.4765,.4102],[.0007,.004,.0045])}
obs={"Appearance only (cosine)":([.4650,.4498,.4423,.4389],[0,0,0,0]),"GAT + attention pooling":([.4801,.4634,.4546,.4553],[.006,.0078,.0025,.0064]),
     "Transition-aware GAT + LSTM":([.4558,.4560,.4508,.4490],[.0061,.0038,.0041,.0019]),"Transition-aware GAT + attention (final)":([.4835,.4703,.4582,.4568],[.0007,.0046,.0028,.004])}
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0))
for ax in (a,b): ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
for name,c,m in V:
    y,s=occ[name]; a.errorbar([0,1,2],y,yerr=s,fmt="-"+m,color=c,lw=2,ms=5,capsize=2,label=name)
    y,s=obs[name]; b.errorbar([0,1,2,3],y,yerr=s,fmt="-"+m,color=c,lw=2,ms=5,capsize=2,label=name)
a.set_xticks([0,1,2]); a.set_xticklabels(["clean","occ30","occ50"]); a.set_ylabel("IDF1 (S02+S05)"); a.set_title("(a) partial occlusion",loc="left",fontsize=9,color=INK)
b.set_xticks([0,1,2,3]); b.set_xticklabels(["100%","50%","25%","10%"]); b.set_xlabel("fraction of each tracklet observed"); b.set_title("(b) partial observation",loc="left",fontsize=9,color=INK)
h,l=a.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig("fig_robust.png",dpi=300); plt.close(fig)
print("ok")
