import matplotlib.pyplot as plt

rmseb = calc_rmse(Epol_clust_no_el,Epol_pair_no_el)
slopeb, interceptb, r_valueb, p_valueb, std_errb = scipy.stats.linregress(Epol_clust_no_el, Epol_pair_no_el)

gmin=min(np.min(Epol_clust_no_el),np.min(Epol_pair_no_el))
gmax=max(np.max(Epol_clust_no_el),np.max(Epol_pair_no_el))

cols=['blue','red','green','orange','brown']
axisFont = {'family' : 'sans', 'weight' : 'bold', 'size'   : 22}
font1 = {'size':15}
font2 = {'size':14}
diax=[gmin,gmax]
diay=[gmin,gmax]

fig, ax = plt.subplots(figsize=(7, 7))

ax.tick_params(axis='both', which='major', labelsize=20)
plt.scatter(Epol_clust_no_el, Epol_pair_no_el, c=cols[0], alpha=0.4)

plt.plot(diax,diay,c='gray')
#ax.legend(loc='upper left')
plt.xlabel('Epol cluster (kcal/mol)',fontdict=axisFont)
plt.ylabel('Epol dimers (kcal/mol)',fontdict=axisFont)
#plt.title('Total polarization interaction energy without Eel',fontdict=font1)
plt.text(-18,-6, 'H2O RMSE: %.2f'%rmseb +'\nr2: %.2f'%r_valueb, fontsize=20,color=cols[0])
plt.xlim([gmin, gmax])
plt.ylim([gmin, gmax])

fname='/group-data/devereux/CHARMM-EDA/figs/wat_pol_vs_2body.pdf'
plt.savefig(fname, format='pdf',bbox_inches='tight')

plt.show()