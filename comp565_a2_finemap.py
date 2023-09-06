# -*- coding: utf-8 -*-
"""COMP565_A2_finemap

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FZUgIh4v0ONSNOQvNTqSn-E5TMf4mhdn
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from itertools import combinations
import matplotlib.pyplot as plt

"""Load the data"""

z = pd.read_csv("/content/gdrive/My Drive/COMP 565/A2/zscore.csv")
LD= pd.read_csv("/content/gdrive/My Drive/COMP 565/A2/LD.csv")
SNP_pip = pd.read_csv("/content/gdrive/My Drive/COMP 565/A2/SNP_pip.csv")

"""We assume there is maximum 3 casual SNPs"""

#z # 100*2

#LD #100*101

#SNP_pip #100*2

LD_matrix = LD.drop('Unnamed: 0', axis = 1)
LD_matrix.index = LD_matrix.columns

z_score = z.drop('Unnamed: 0', axis = 1)
z_score.index = LD_matrix.columns
SNP_PIP = SNP_pip.drop('Unnamed: 0', axis = 1)

#z_score

#LD_matrix # 100*100

np.fill_diagonal(LD_matrix.values, 1) # the correlation with itself to 1

#LD_matrix

#SNP_PIP # 100*1

"""Q1.We first implement the efficient Bayes factor for each causal configurations"""

snp_names = LD_matrix.index.tolist()

comb1_list = []
for x in combinations(snp_names, 1):
   comb1_list.append(x)
one_SNP = pd.DataFrame(comb1_list, columns=["SNP1"])

comb2_list = []
for y in combinations(snp_names, 2):
   comb2_list.append(y)
two_SNP = pd.DataFrame(comb2_list, columns=["SNP1","SNP2"])

comb3_list = []
for z in combinations(snp_names, 3):
   comb3_list.append(z)
three_SNP = pd.DataFrame(comb3_list, columns=["SNP1","SNP2","SNP3"])

def calBF(rcc, z_score,number_of_cSNP):
  
   sigma_cc = 2.49 * np.identity(number_of_cSNP)
   zero = np.zeros(number_of_cSNP)
   cov_num = rcc + np.matmul(rcc, np.matmul(sigma_cc,rcc))
  
   numerator = multivariate_normal.pdf(z_score,mean= zero,cov=cov_num)
   denominator = multivariate_normal.pdf(z_score,mean= zero,cov=rcc)
   bf = numerator/denominator
   return bf

def BF(num_of_cSNP =1):
  if num_of_cSNP == 1:
    BF1= []
    for index, rows in one_SNP.iterrows():
        snp1 = rows.SNP1
        z_score1 = z_score[z_score.index == snp1]["V1"].to_numpy()

        LD1 = LD_matrix[snp1]
        rcc1 = LD1.loc[one_SNP.iloc[index]].to_numpy()

        bf1 = calBF(rcc1, z_score1, 1)
        BF1.append(bf1)
    return BF1
  

  elif num_of_cSNP == 2:
    BF2= []
    for index, rows in two_SNP.iterrows():
        snp1 = rows.SNP1
        snp2 = rows.SNP2
        z_score2 = z_score[(z_score.index).isin([snp1,snp2])]["V1"].to_numpy()

        LD2 = LD_matrix[[snp1,snp2]]
        rcc2 = LD2.loc[two_SNP.iloc[index]].to_numpy()
        
        bf2 = calBF(rcc2, z_score2, 2)
      
        BF2.append(bf2)
    return BF2
  
  else:
    BF3= []
    for index, rows in three_SNP.iterrows():

        snp1 = rows.SNP1
        snp2 = rows.SNP2
        snp3 = rows.SNP3
        z_score3 = z_score[(z_score.index).isin([snp1,snp2,snp3])]["V1"].to_numpy()

        LD3 = LD_matrix[[snp1,snp2,snp3]]
        rcc3 = LD3.loc[three_SNP.iloc[index]].to_numpy()
        
        bf3 = calBF(rcc3, z_score3, 3)
        BF3.append(bf3)
    return BF3

#BF(num_of_cSNP = 1)

one_cSNP = one_SNP.copy()
two_cSNP = two_SNP.copy()
three_cSNP = three_SNP.copy()

one_cSNP['BF']= BF(num_of_cSNP = 1)
two_cSNP['BF'] = BF(num_of_cSNP = 2)
three_cSNP['BF'] = BF(num_of_cSNP = 3)

"""Q2: Implement the prior calculation"""

# number of SNPs
m = 100
#calculate the prior
one_cSNP['prior'] = ((1/m)**1) * ((m-1)/m)**(m-1)
two_cSNP['prior'] = ((1/m)**2) * ((m-1)/m)**(m-2)
three_cSNP['prior'] = ((1/m)**3) * ((m-1)/m)**(m-3)

"""Q3 Implement posterior inference"""

#calculate posterior
all_configs = pd.concat([one_cSNP, two_cSNP, three_cSNP])
all_configs=all_configs[['SNP1','SNP2','SNP3','BF','prior']]

all_configs["posterior"] = (all_configs['BF']*all_configs['prior'])/ ((all_configs['BF']*all_configs['prior']).sum())
#all_configs

# sort by increasing order of posterior
all_configs = all_configs.sort_values(by=['posterior'])
all_configs.reset_index(drop=True, inplace=True)
all_configs['sorted_configurations'] = all_configs.index

#all_configs

fig2=all_configs.plot.scatter(x="sorted_configurations", y="posterior",c='black', alpha=1)
plt.title("Posteriors of all of the valid configurations in increasing order")
plt.show()

#create pd to store SNP-level PIP
snp_pprob = pd.DataFrame(data = snp_names, columns = ['SNP_names'])
all_configs_sum = all_configs["posterior"].sum()
pprob = []
for x in snp_names:
    # all combinations that contain snp "x"
    snp_config = all_configs[(all_configs['SNP1'] == x) | (all_configs['SNP2'] == x)| (all_configs['SNP3'] == x)]
    snp_sum = snp_config["posterior"].sum() 
    x_pprob = snp_sum/all_configs_sum
    pprob.append(x_pprob)
    
snp_pprob['SNP_pip'] = pprob

#download the file
from google.colab import files
snp_pprob.to_csv('COMP565_A2_SNP_pip.csv.gz', encoding = 'utf-8-sig') 
files.download('COMP565_A2_SNP_pip.csv.gz')

#compare with given data
#diff = pd.DataFrame(data = (SNP_PIP['x']- snp_pprob["SNP_pip"]), columns = ['difference between given and calculated'])
#diff

"""Graph"""

snp_pprob2 = snp_pprob.copy()
snp_pprob2.index = snp_names
# snp_pprob2

# calculate -log10 p-values with marginal z score
z_score['p_value'] = scipy.stats.norm.sf(abs(z_score['V1']))
z_score['-log10p'] = - np.log10(z_score['p_value'])
#add SNP_names in order to merge according to snp names
z_score ['SNP_names'] = z_score.index
result = pd.merge(z_score, snp_pprob, on = ['SNP_names'])

fig, (ax1, ax2) = plt.subplots(2, 1)  # 1 row, 2 columns
scatter1 = ax1.scatter(x= z_score ['SNP_names'], y = z_score['-log10p'])
scatter2 = ax2.scatter(x= snp_pprob['SNP_names'], y=snp_pprob['SNP_pip'])
ax1.set_ylabel('-log10p')
ax2.set_ylabel('SNP_pip')
ax1.scatter(x= ['rs10104559','rs1365732','rs12676370'], y = [z_score.loc['rs10104559']['-log10p'],z_score.loc['rs1365732']['-log10p'],z_score.loc['rs12676370']['-log10p']],c ='red')
ax2.scatter(x= ['rs10104559','rs1365732','rs12676370'], y = [snp_pprob2.loc['rs10104559']['SNP_pip'],snp_pprob2.loc['rs1365732']['SNP_pip'],snp_pprob2.loc['rs12676370']['SNP_pip']],c ='red')
# remove x-axis labels
ax1.axes.get_xaxis().set_visible(False)
ax2.axes.get_xaxis().set_visible(False)


plt.show()