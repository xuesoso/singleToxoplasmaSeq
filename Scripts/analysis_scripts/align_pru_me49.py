import scanpy.api as sc
from umap import UMAP
import scanorama
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(script_path, '../../Figures') + '/'
adata_scv_pru = sc.read_h5ad(output_dir+'../Data/pru/adata_sc_velocyto.h5ad')
adata_scv_me49 = sc.read_h5ad(output_dir+'../Data/011_me49/adata_sc_velocyto.h5ad')

adatas = [adata_scv_me49.copy(), adata_scv_pru.copy()]
integrated, corrected = scanorama.correct_scanpy(adatas, return_dimred=True)
merged_x = np.concatenate(integrated)
umap_merged_x = UMAP(n_components=2, random_state=4, min_dist=0.3,
                     n_neighbors=50).fit_transform(merged_x)
adatas = corrected[0].concatenate(corrected[1])
adatas.obs_names = [x.split('-')[0] for x in adatas.obs_names]
adatas.obsm['X_corrected'] = merged_x
adatas.obsm['X_corrected_umap'] = umap_merged_x
adatas.layers['original_mat'] = sp.sparse.csr_matrix(np.concatenate([
    adata_scv_me49.X.A, adata_scv_pru.X.A]))
batch = ['ME49' if '10099011' in x else 'Pru' for x in adatas.obs_names]
adatas.obs['batch'] = batch

## Save scanorama results
adatas.write_h5ad(filename=output_dir+
                  '../Data/pru/adata_integrated_0506_me49.h5ad',
                 compression='gzip')
