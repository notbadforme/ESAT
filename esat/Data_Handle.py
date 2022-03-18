import os
import glob
import pandas as pd
import numpy as np

csv_url="nlst_path_prsn_20180727.csv"
svs_url="../NLST2/"
df = pd.read_csv(csv_url)[["pid","death_days_2015"]]

svs_file=glob.glob(svs_url+'*/*.svs')
for i in range(len(df)):
    svs_file=glob.glob(svs_url+str(int(df.loc[i][0]))+'/*.svs')
    surv=df.loc[i][1]
    if np.isnan(surv):
        status=0
        surv=-1
    else:
        status=1
        surv=int(surv)
    for j in range(len(svs_file)):
        os.rename(svs_file[j],svs_url+str(int(df.loc[i][0]))+'/'+str(j)+'_'+str(surv)+'_'+str(status)+'.svs')


