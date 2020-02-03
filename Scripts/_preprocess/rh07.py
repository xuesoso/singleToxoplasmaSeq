from _loadlib.rh07 import *

## Preprocess data

#```python
#### We are loading and using the union aligned Pru datasets
df_id = pd.read_csv(gene_dictionary_input, index_col=0, sep='\t')
gene_to_product = {x:y for x,y in zip(df_id['gene'].values, df_id['product'].values)};
data = sat.pp.convert_id_to_gene(sat.pp.load_dataframes(htseq_input,\
                                                        remove_last_nrows=5).T, df_id, start='id')
adata = anndata.AnnData(data)
adata.X = sp.sparse.csr_matrix(adata.X)
adata.obs.index = ['_'.join(x.split('_')[:3]) for x in adata.obs_names]
#```

# If you have loom file that stores RNA velocyto output, we convert .loom to .h5ad first, otherwise errors arise in scvelo

#```python
adata_loom = scv.read(input_folder_dir+'07_RH_velocyto.loom')
adata_rv = sat.loom_to_h5ad(adata_loom)
adata_loom = None
sc.write(input_folder_dir+'07_RH_velocyto.h5ad', adata_rv)
#```

#```python
#### Process adata and add meta annotation information
df_star = sat.pp.load_dfStar(star_input)
adata.var['product'] = sat.get_product(adata.var_names.values, gene_to_product)
for key in df_star.columns.values:
    adata.obs[key] = df_star.loc[adata.obs_names, key]
adata.obs = adata.obs.fillna(0)
#### Load Velocyto h5ad
adata_rv = sc.read_h5ad(input_folder_dir+'07_RH_velocyto.h5ad')
intersect = np.intersect1d(adata_rv.obs_names.values, adata.obs_names.values)
adata_rv = adata_rv[intersect][:, adata.var_names]
adata = adata[intersect]
#### copy over Velocyto information to adata
for key in adata_rv.var.columns.values:
    adata.var[key] = adata_rv.var[key].copy()
for key in adata_rv.layers.keys():
    adata.layers[key] = adata_rv.layers[key].copy()
#```

#```python
sc.write(output_dir+'adata_rv.h5ad', adata_rv)
sc.write(output_dir+'adata_raw.h5ad', adata)
#```

### Filter cells

#```python
adata_raw = sc.read_h5ad(output_dir+'adata_raw.h5ad')
gene_to_product = {x:y for x,y in zip(adata_raw.var.index.values, adata_raw.var['product'].values)};
ribosomalRNA = 'TGGT1_412150'
ercc = sat.get_gene('ERCC-', gene_to_product)['gene'].values
read_sum = adata_raw.X.sum(axis=1).A.flatten()
gene_count = np.greater(adata_raw.X.A, 0).sum(axis=1)
per_rrna = (adata_raw[:, ribosomalRNA].X / adata_raw.X.sum(axis=1).flatten()).\
A.flatten()*100
per_ercc = (adata_raw[:, ercc].X.sum(axis=1) / adata_raw.X.sum(axis=1)).A.flatten()*100
#```

#```python
plt.hist(per_rrna, bins=20);
plt.title('% rRNA');
#```

#```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            per_rrna, s=5, c='', alpha=0.5, edgecolor='steelblue');
max_rrna = 0.1
plt.axhline(max_rrna, ls='--', c='k');
plt.ylabel('% rRNA'); plt.xlabel('Read sum');
fig.savefig(output_dir+'rs_vs_rrna.pdf', bbox_inches='tight')
#```

#```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            gene_count, s=5, c='', alpha=0.5, edgecolor='steelblue');
plt.xscale('log')
max_gc = 2500
min_gc = 600
min_read = 5*10**4
plt.axhline(min_gc, ls='--', c='k');
plt.axhline(max_gc, ls='--', c='k');
plt.axvline(min_read, ls='--', c='k');
plt.ylabel('Gene count'); plt.xlabel('Read sum');
fig.savefig(output_dir+'rs_vs_gc.pdf', bbox_inches='tight')
#```

#```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            per_ercc, s=5, c='', alpha=0.5, edgecolor='steelblue');
plt.xscale('log')
min_read = 5*10**4
max_ercc = 15
plt.axhline(max_ercc, ls='--', c='k');
plt.axvline(min_read, ls='--', c='k');
plt.ylabel('% ERCC'); plt.xlabel('Read sum');
fig.savefig(output_dir+'rs_vs_ercc.pdf', bbox_inches='tight')
#```

#```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            adata_raw.obs['percent_mapped'],\
            s=5, c='', alpha=0.5, edgecolor='steelblue');
plt.xscale('log')
min_read = 5*10**4
min_per_mapped = 20
plt.axhline(min_per_mapped, ls='--', c='k');
plt.axvline(min_read, ls='--', c='k');
plt.ylabel('% mapped'); plt.xlabel('Read sum'); plt.legend();
fig.savefig(output_dir+'rs_vs_mapped.pdf', bbox_inches='tight')
#```

#### Threshold for cell quality

#```python
max_gc = 2500
min_gc = 600
min_read = 5*10**4
max_ercc = 15
max_rrna = 0.1
min_per_mapped = 20
subset = np.repeat(True, adata_raw.shape[0])
subset[per_rrna >= max_rrna] = False
subset[per_ercc >= max_ercc] = False
subset[gene_count <= min_gc] = False
subset[gene_count >= max_gc] = False
subset[read_sum <= min_read] = False
subset[adata_raw.obs['percent_mapped'].values <= min_per_mapped] = False
#```


#```python
adata_allg = adata_raw[subset].copy()
adata_allg.obs['gene_count'] = gene_count[subset]
adata_allg.obs['read_sum'] = read_sum[subset]
adata_allg.X = sp.sparse.csr_matrix(sat.cpm_normalize(adata_allg.X.A))
sc.write(output_dir+'adata_allg.h5ad', adata_allg)
#```

#```python
subset = np.array([True if x in adata_allg.obs_names else False for x in
          adata_raw.obs_names])
rs_filtered = adata_raw[subset].X.A.sum(axis=1)
gc_filtered = np.greater(adata_raw[subset].X.A, 0).sum(axis=1)
rs_analyzed = adata_raw[subset != True].X.A.sum(axis=1)
gc_analyzed = np.greater(adata_raw[subset != True].X.A, 0).sum(axis=1)
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(rs_filtered, gc_filtered, c='xkcd:blue', label='Analyzed', s=5)
ax.scatter(rs_analyzed, gc_analyzed, c='xkcd:crimson', label='Filtered', s=5)
ax.set_xscale('log')
ax.set_xlim(10**2,)
ax.set_xlabel('Mapped read', fontsize=12)
ax.set_ylabel('Gene count', fontsize=12)
ax.set_title('Gene detection', fontsize=12)
ax.legend(frameon=False, loc=(0.05, 0.7))
fig.savefig(output_dir+'gc_rs_filter.pdf', bbox_inches='tight',
            transparent=True)
#```

### Filter genes


# Now we remove ribosomal RNA, ERCC and transgenes genes from the dataset

#```python
adata_allg = sc.read_h5ad(output_dir+'adata_allg.h5ad')
transgenes = ['transgene:GFP', 'transgene:TdTomato']
keep_genes = np.setdiff1d(adata_allg.var.index.values,
                          np.concatenate([ercc, [ribosomalRNA], transgenes]))
adata = adata_allg[:, keep_genes].copy()
adata.X = sp.sparse.csr_matrix(sat.cpm_normalize(adata.X.A))
adata.obs['gene_count'] = np.greater(adata.X.A, 0).sum(axis=1)
sc.write(output_dir+'adata_filtered.h5ad', adata)
#```
