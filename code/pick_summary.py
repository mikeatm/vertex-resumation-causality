import json, numpy as np, sys
ns={'dimer':2,'chain3':3,'ring3':3,'ring4':4}
scan={(r['cluster'],r['beta'],r['U'],r['mu']):r for r in map(json.loads,open('splice_scan.jsonl')) if r['factor']==2}
R=[json.loads(l) for l in open('indep_pick.jsonl')]
R=[r for r in R if 0.1<scan[(r['cluster'],r['beta'],r['U'],r['mu'])]['N']<2*ns[r['cluster']]-0.1]
NP=[2,3,4,6,8]; TH=1e-9
print('points',len(R))
print('controls: min scaled Pick  ex',min(min(r['ex']) for r in R),' gw',min(min(r['gw']) for r in R))
for cl in ['dimer','chain3','ring3','ring4','ALL']:
    P=[r for r in R if cl=='ALL' or r['cluster']==cl]
    if not P: continue
    row=[sum(r['gd'][j]<-TH for r in P) for j in range(5)]
    rowG=[sum(r['Ggd'][j]<-TH for r in P) for j in range(5)]
    pw=sum(scan[(r['cluster'],r['beta'],r['U'],r['mu'])]['v_gd']>1e-6 for r in P)
    pwG=sum(min(r['lam_gd'])<-1e-6 for r in P)
    print(f"{cl:6s} n={len(P):3d} pointwise={pw:3d} (indep {pwG})  Pick Sigma NP{NP}={row}  Pick G={rowG}  mu0 Pick8={sum(r['gd'][4]<-TH for r in P if r['mu']==0)}/{sum(1 for r in P if r['mu']==0)}")
# weakest Sigma failures at NP=8 (margin above roundoff)
fails=sorted([r['gd'][4] for r in R if r['gd'][4]<-TH])
print('weakest Sigma Pick8 failures', fails[-5:] if fails else None)
ok=[r for r in R if r['gd'][4]>=-TH]
print('passing at NP=8:',[(r['cluster'],r['beta'],r['U'],r['mu'],'%.1e'%r['gd'][4]) for r in ok][:12])
