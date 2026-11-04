import h5py, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
p="/pfs/10/work/hd_id260-fno_training/data/compressed/loreli_slices_l294_v3.h5"
h=h5py.File(p,"r")
off,scale=json.loads(h.attrs["tb_norm"]); pad=int(h.attrs["pad"]); n=int(h.attrs["n_out"]); PAD=pad; pad=0
bands=json.loads(h.attrs["bands"])
cone=h["cone_id"][:]; xm=h["xhi_mean"][:]; bg=h["background"][:]; z=h["z"][:]; split=h["split"][:]; los=h["los_index"][:]
C=12568
rows=np.flatnonzero(cone==C)
pick=[]
for t in (0.85,0.5,0.2):
    act=rows[~bg[rows]]; pick.append(act[np.argmin(np.abs(xm[act]-t))])
pick.append(rows[bg[rows]][0])
pick=sorted(pick,key=lambda r:-xm[r])
d0=bands.index([0,1]); v0=13+d0
cols=[("target x_HI","x",None),("T_b [mK]","tb",None),(f"density band {bands[d0]} (cond ch {d0})","c",d0),(f"LOS velocity band {bands[d0]} (cond ch {v0})","c",v0),("phys A(z)(1+δ) [mK]","phys",None)]
fig,axes=plt.subplots(len(pick),len(cols),figsize=(4*len(cols),3.6*len(pick)),constrained_layout=True)
for i,r in enumerate(pick):
    sl=np.s_[PAD:PAD+n,PAD:PAD+n]; tgt=h["target"][r][(slice(None),)+sl].astype(np.float32); cond=h["cond"][r][(slice(None),)+sl].astype(np.float32); ph=h["phys"][r][sl]
    for j,(title,kind,ch) in enumerate(cols):
        ax=axes[i,j]
        if kind=="x": im=ax.imshow(tgt[0],cmap="Greys_r",vmin=0,vmax=1)
        elif kind=="tb":
            tb=tgt[1]*scale+off; inner=tb[pad:pad+n,pad:pad+n]
            im=ax.imshow(tb,cmap="RdBu_r",vmin=min(inner.min(),-1),vmax=max(inner.max(),1)) if False else ax.imshow(tb,cmap="magma",vmin=np.percentile(inner,1),vmax=np.percentile(inner,99.5))
        elif kind=="c":
            a=cond[ch]; inner=a[pad:pad+n,pad:pad+n]; s=np.percentile(np.abs(inner),99.5)
            im=ax.imshow(a,cmap="RdBu_r",vmin=-s,vmax=s)
        else:
            inner=ph[pad:pad+n,pad:pad+n]; im=ax.imshow(ph,cmap="viridis",vmin=np.percentile(inner,1),vmax=np.percentile(inner,99.5))
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=ax,shrink=0.8)
        if i==0: ax.set_title(title,fontsize=11)
        if j==0:
            tag="background" if bg[r] else "active"
            ax.set_ylabel(f"row {r}  los {los[r]}\nz={z[r]:.2f}  <x_HI>={xm[r]:.2f}\n{tag}",fontsize=10)
fig.suptitle(f"LoReLi cache loreli_slices_l294_v3.h5 — sim {C} (train, block 428). valid 206×206 interior only (294 Mpc, 1.43 Mpc/cell).",fontsize=12)
out=sys.argv[1]
fig.savefig(out+".png",dpi=200); fig.savefig(out+".pdf"); print(out, [ (int(r), round(float(z[r]),2), round(float(xm[r]),3)) for r in pick])
