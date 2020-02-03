from _loadlib.me49_011 import *

## Preprocess data

# ```python
#### We are loading and using the union aligned Pru datasets
df_id = pd.read_csv(gene_dictionary_input, index_col=0, sep='\t')
gene_to_product = {x:y for x,y in zip(df_id['gene'].values, df_id['product'].values)};
data = sat.pp.convert_id_to_gene(sat.pp.load_dataframes(htseq_input,\
                                    remove_last_nrows=5).T, df_id, start='id')
adata = anndata.AnnData(data)
adata.X = sp.sparse.csr_matrix(adata.X)
adata.obs.index = ['_'.join(x.split('_')[:3]) for x in adata.obs_names]
# ```

# If you have loom file that stores RNA velocyto output, we convert .loom to .h5ad first, otherwise errors arise in scvelo

# ```python
adata_loom = scv.read(input_folder_dir+'011_me49_velocyto.loom')
adata_rv = sat.loom_to_h5ad(adata_loom)
adata_loom = None
sc.write(input_folder_dir+'011_me49_velocyto.h5ad', adata_rv)
# ```

# ```python
#### Process adata and add meta annotation information
df_star = sat.pp.load_dfStar(star_input)
df_star.index = ['_'.join(x.split('_')[:3]) for x in df_star.index.values]
adata.var['product'] = sat.get_product(adata.var_names.values, gene_to_product)
adata.var.index = [x.replace('trans', 'transgene:') if 'transGFP' in x or 'transTdTomato' in x \
                   else x for x in adata.var_names]
for key in df_star.columns.values:
    adata.obs[key] = df_star.loc[adata.obs_names, key]
adata.obs = adata.obs.fillna(0)
dpi = {'1009901101':'Day 3', '1009901102':'Day 3', '1009901103':'Day 3', '1009901104':'Day 3',\
      '1009901105':'Day 3', '1009901106':'Day 0'}
adata.obs['dpi'] = [dpi[x.split('_')[1]] for x in adata.obs_names]
#### Load Velocyto h5ad
adata_rv = sc.read_h5ad(input_folder_dir+'011_me49_velocyto.h5ad')
intersect = np.intersect1d(adata_rv.obs_names.values, adata.obs_names.values)
adata_rv = adata_rv[intersect][:, adata.var_names]
adata = adata[intersect]
#### copy over Velocyto information to adata
for key in adata_rv.var.columns.values:
    adata.var[key] = adata_rv.var[key].copy()
for key in adata_rv.layers.keys():
    adata.layers[key] = adata_rv.layers[key].copy()
# ```

# ```python
sc.write(output_dir+'adata_rv.h5ad', adata_rv)
sc.write(output_dir+'adata_raw.h5ad', adata)
# ```

### Filter cells

# ```python
adata_raw = sc.read_h5ad(output_dir+'adata_raw.h5ad')
gene_to_product = {x:y for x,y in zip(adata_raw.var.index.values, adata_raw.var['product'].values)};
ribosomalRNA = sat.get_gene('ribosomal RNA', gene_to_product)['gene'].values
ribosomalRNA = np.array([x for x, y in zip(ribosomalRNA,
                   sat.get_product(ribosomalRNA, gene_to_product))
                         if 'ribosomal RNA ' not in y])
ercc = sat.get_gene('ERCC-', gene_to_product)['gene'].values
read_sum = adata_raw.X.sum(axis=1).A.flatten()
gene_count = np.greater(adata_raw.X.A, 0).sum(axis=1)
per_rrna = (adata_raw[:, ribosomalRNA].X.sum(axis=1) / adata_raw.X.sum(axis=1)).A.flatten()*100
per_ercc = (adata_raw[:, ercc].X.sum(axis=1) / adata_raw.X.sum(axis=1)).A.flatten()*100
# ```

# ```python
plt.hist(per_rrna, bins=50);
plt.title('% rRNA');
# ```

# ```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            per_rrna, s=5, c='', alpha=0.5, edgecolor='steelblue');
max_rrna = 4
plt.axhline(max_rrna, ls='--', c='k');
plt.ylabel('% rRNA'); plt.xlabel('Read sum');
fig.savefig(output_dir+'rs_vs_rna.pdf', bbox_inches='tight')
# ```

# ```python
fig, ax = plt.subplots(figsize=(3, 3))
plt.scatter(read_sum, \
            gene_count, s=5, c='', alpha=0.5, edgecolor='steelblue');
plt.xscale('log')
max_gc = 4000
min_gc = 600
min_read = 5*10**4
plt.axhline(min_gc, ls='--', c='k');
plt.axhline(max_gc, ls='--', c='k');
plt.axvline(min_read, ls='--', c='k');
plt.ylabel('Gene count'); plt.xlabel('Read sum');
fig.savefig(output_dir+'rs_vs_gc.pdf', bbox_inches='tight')
# ```

# ```python
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
# ```

# ```python
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
# ```

#### Threshold for cell quality

# ```python
max_gc = 4000
min_gc = 600
min_read = 5*10**4
max_ercc = 15
max_rrna = 4
min_per_mapped = 20
subset = np.repeat(True, adata_raw.shape[0])
subset[per_rrna >= max_rrna] = False
subset[per_ercc >= max_ercc] = False
subset[gene_count <= min_gc] = False
subset[gene_count >= max_gc] = False
subset[read_sum <= min_read] = False
subset[adata_raw.obs['percent_mapped'].values <= min_per_mapped] = False
# ```

# ```python
adata_allg = adata_raw[subset].copy()
adata_allg.obs['gene_count'] = gene_count[subset]
adata_allg.obs['read_sum'] = read_sum[subset]
adata_allg.X = sp.sparse.csr_matrix(sat.cpm_normalize(adata_allg.X.A))
sc.write(output_dir+'adata_allg.h5ad', adata_allg)
# ```

### Filter genes

# Now we remove ribosomal RNA, ERCC and transgenes genes from the dataset

# ```python
adata_allg = sc.read_h5ad(output_dir+'adata_allg.h5ad')
transgenes = ['transgene:GFP', 'transgene:TdTomato']
keep_genes = np.setdiff1d(adata_allg.var.index.values, np.concatenate([ercc, ribosomalRNA, transgenes]))
adata = adata_allg[:, keep_genes].copy()
adata.X = sp.sparse.csr_matrix(sat.cpm_normalize(adata.X.A))
adata.obs['gene_count'] = np.greater(adata.X.A, 0).sum(axis=1)
sc.write(output_dir+'adata_filtered.h5ad', adata)
# ```

