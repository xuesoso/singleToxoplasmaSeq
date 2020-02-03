import sys
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import sc_tools as sat

def rna_velocity(adata_in, n_neighbors=20, top_genes=2000, mode='stochastic', \
                 dimensions=2, batch_labels = [], basis='umap', \
                 subset_genes=[], **args):
    import scvelo as scv
    import scanpy as sc
    scv.pp.show_proportions(adata_in)
    if len(subset_genes) > 0:
        adata_copy = adata_in.copy()[:, subset_genes]
        sc.pp.normalize_per_cell(adata_copy)
        print('Subsetting the adata by the provided subset_genes')
    else:
        adata_copy = adata_in.copy()
        scv.pp.filter_and_normalize(adata_copy, log=False, min_counts=3,
                                    min_counts_u=3, n_top_genes=top_genes)
    sc.pp.log1p(adata_copy)
    sc.pp.scale(adata_copy, max_value=5)
    if len(batch_labels) > 0:
        adata_corrected = sc.pp.mnn_correct(adata_in[adata_in.obs['batch'] ==
                                                     batch_labels[0]],
                                            adata_in[adata_in.obs['batch'] ==
                                                     batch_labels[1]],
                                            batch_categories=batch_labels,
                                            do_concatenate=True)[0]
        adata_corrected.obs.index = [x.split('-')[0] for x in
                                     adata_corrected.obs_names.values]
        adata_in.X = adata_corrected[adata_in.obs_names.values].X
    scv.pp.pca(adata_copy)
    scv.pp.neighbors(adata_copy, metric='correlation', knn=n_neighbors)
    scv.tl.umap(adata_copy, random_state=4, n_components=dimensions, **args)
    adata_in.obsm = adata_copy.obsm
    adata_in.uns = adata_copy.uns
    adata_in.var['used_for_projection'] = np.zeros(adata_in.shape[1],
                                                   dtype=bool)
    adata_in.var.loc[adata_copy.var.index, 'used_for_projection'] = True
    scv.pp.moments(adata_in, metric='correlation', n_neighbors=n_neighbors)
    scv.tl.velocity(adata_in, mode=mode)
    scv.tl.velocity_graph(adata_in, n_neighbors=n_neighbors, basis=basis)
    scv.tl.velocity_embedding(adata_in)

def group_cells(adata_in, resolution=1, eps=0.1, random_state=4):
    import scvelo as scv
    scv.tl.louvain(adata_in, resolution=resolution, random_state=random_state,
                   directed=True)
    scv.tl.terminal_states(adata_in, basis='X_umap', eps=eps)
    adata_in.var['velocity_genes'] = adata_in.var['velocity_genes'].astype(bool)
    scv.tl.rank_velocity_genes(adata_in, match_with='louvain', resolution=.8)

def update_gene_count(X, min_exp=1):
    if sp.sparse.issparse(X.X):
        gc = np.greater(X.X.A, min_exp).sum(axis=1)
    else:
        gc = np.greater(X.X, min_exp).sum(axis=1)
    X.obs['gene_count'] = gc

def get_gc_rrna(X, min_exp=0):
    """
    Take in anndata object, and then calculates gene count and percent rRNA
    """
    ribosomal = [x for x, y in zip(X.var.index.values, X.var['product'].values)\
                 if 'ribosomal RNA' in y and 'ribosomal RNA ' not in y]
    ribosomal.append('TGGT1_412150')
    ercc_ind = [x for x in X.var.index.values if 'ERCC-' in x]
    keep_genes = np.setdiff1d(X.var.index.values, np.concatenate([ribosomal, ercc_ind]))
    gene_count = np.greater(X[:, keep_genes].X.A, min_exp).sum(axis=1).\
            flatten()
    percent_rrna = (X[:, ribosomal].X.A.sum(axis=1) /
                    X.X.A.sum(axis=1)).flatten()
    return (gene_count, percent_rrna)

class dtest(object):
    def __init__(self, data, cluster_labels, cluster_of_interest=1, minimum_fold=None):
        self.ndim = np.ndim(data)
        self.minimum_fold = minimum_fold
        if isinstance(data, pd.DataFrame):
            self.features = data.columns.values
        elif isinstance(data, pd.Series):
            self.features = np.array([data._name])
        else:
            if self.ndim == 1:
                self.features = np.array([0])
            else:
                self.features = np.arange(np.shape(data)[1])
        self.data = np.array(data).copy()
        self.cluster_labels = cluster_labels
        assert np.shape(cluster_labels)[0] == np.shape(self.data)[0], 'The length of data matrix must match the length of provided clusters'
        self.cluster_of_interest = cluster_of_interest
        assert cluster_of_interest in self.cluster_labels, 'The provided cluster_of_interest label must be in the cluster_labels array'
        self.sample_size = np.shape(data)[0]

    def hypothesis_test(self, method='wilcoxon', correction='bh'):
        available_methods = ['wilcoxon', 'mannwhitneyu', 'ttest', 'kruskal']
        assert method in available_methods, 'Provided method is not available'
        test_group = np.zeros(self.sample_size, dtype=bool)
        test_group[self.cluster_labels == self.cluster_of_interest] = True
        self._choose_test(self.data, method=method, test_group=test_group)
        out = pd.DataFrame(0, index=self.features, columns=['fold', 'log2-fold', 'p', 'adjusted-p'])
        out.loc[self.features, ['fold']], out.loc[self.features, ['p']] = \
                self.fold, self.p
        out['log2-fold'] = np.log2(out['fold'].values)
        if method == 'kruskal':
            out['max_cluster'] = 0
            out.loc[self.features, ['max_cluster']] = self.max_cluster
        if self.minimum_fold:
            out = out.loc[out['log2-fold'].abs() > self.minimum_fold]
        out = out.sort_values('p')
        out['adjusted-p'] = self._multiple_test_correction(out['p'].values, correction=correction)
        return out

    def _choose_test(self, D, method, test_group):
        if method != 'kruskal':
            self.fold = D[test_group].mean(axis=0) / D[test_group == False].mean(axis=0)
        else:
            self.fold = []
            self.max_cluster = []
        if self.ndim == 1:
            if method == 'mannwhitneyu':
                _stats, self.p = sp.stats.mannwhitneyu(D[test_group], D[test_group == False], alternative='two-sided')
            elif method == 'wilcoxon':
                _stats, self.p = sp.stats.ranksums(D[test_group], D[test_group == False])
            elif method == 'ttest':
                _stats, self.p = sp.stats.ttest_ind(D[test_group], D[test_group == False])
            elif method == 'kruskal':
                unique_cluster_labels = np.unique(self.cluster_labels)
                D, mean_v = self.separate_by_clusters(D, self.cluster_labels)
                _stats, self.p = sp.stats.kruskal(*D)
                max_ind = np.argmax(mean_v)
                mask = np.ones(np.shape(mean_v)[0], dtype=bool)
                mask[max_ind] = False
                self.fold = mean_v[max_ind] / np.mean(mean_v[mask])
                self.max_cluster.append(unique_cluster_labels[max_ind])
        else:
            self.p = []
            for exp in D.T:
                if method == 'mannwhitneyu':
                    _stats, p_val = sp.stats.mannwhitneyu(exp[test_group], exp[test_group == False], alternative='two-sided')
                elif method == 'wilcoxon':
                    _stats, p_val = sp.stats.ranksums(exp[test_group], exp[test_group == False])
                elif method == 'ttest':
                    _stats, p_val = sp.stats.ttest_ind(exp[test_group], exp[test_group == False])
                elif method == 'kruskal':
                    unique_cluster_labels = np.unique(self.cluster_labels)
                    exp, mean_v = self.separate_by_clusters(exp, self.cluster_labels)
                    max_value = 0
                    for x in exp:
                        if np.max(x) > max_value:
                            max_value = np.max(x)
                    if max_value == 0:
                        _stats, p_val = np.inf, np.inf
                        self.fold.append(np.nan)
                        self.max_cluster.append('none')
                    else:
                        _stats, p_val = sp.stats.kruskal(*exp)
                        max_ind = np.argmax(mean_v)
                        mask = np.ones(np.shape(mean_v)[0], dtype=bool)
                        mask[max_ind] = False
                        self.fold.append(mean_v[max_ind] / np.mean(mean_v[mask]))
                        self.max_cluster.append(unique_cluster_labels[max_ind])
                self.p.append(p_val)

    def _multiple_test_correction(self, sorted_pvals, correction):
        # Benjamin-Hochberg correction
        if correction == 'bh':
            n = np.shape(sorted_pvals)[0]
            m = np.arange(1, n+1)
            adjusted = sorted_pvals*n/m
        # Westfall-Young correction
        elif correction == 'wy':
            pass
        return adjusted

    def separate_by_clusters(self, vector, clusters):
        out = []
        mean_v = []
        for u in np.unique(clusters):
            out.append(vector[clusters == u])
            mean_v.append(np.mean(vector[clusters == u]))
        return np.array(out), np.array(mean_v)

def return_lower_triangle(ratio_mat, keep_diag = True):
    if keep_diag == True:
        start = 0
    pair_values, pair_names = [], []
    row_names = ratio_mat.index.values
    for eid, ind in enumerate(range(0, np.shape(ratio_mat)[1])):
        curr_col = ratio_mat.columns.values[eid]
        val = ratio_mat.iloc[ind:, eid]
        for x in val.values:
            pair_values.append(x)
        for i in row_names[ind:]:
            pair_names.append([i, curr_col])
    pair_values = np.array(pair_values)
    pair_names = np.array(pair_names)
    return pair_values, pair_names

def pair_expression_binary(binary_mat):
    gene_names = binary_mat.columns.values
    sample_size = binary_mat.shape[0]
    ratio_mat = pd.DataFrame(0, index=gene_names, columns=gene_names)
    for curr_gene in gene_names:
        for target_gene in gene_names:
            if curr_gene == target_gene:
                ratio_mat.loc[curr_gene, target_gene] = 0
            else:
                gene1_only = binary_mat.index[np.logical_and(binary_mat[curr_gene].values > 0,\
                                                            binary_mat[target_gene].values == 0)].shape[0]/sample_size
                gene2_only = binary_mat.index[np.logical_and(binary_mat[curr_gene].values == 0,\
                                                            binary_mat[target_gene].values > 0)].shape[0]/sample_size
                both_on = binary_mat.index[np.logical_and(binary_mat[curr_gene].values > 0,\
                                                         binary_mat[target_gene].values > 0)].shape[0]/sample_size
                ratio = (((gene1_only+both_on)*(gene2_only+both_on))) - both_on
                ratio_mat.loc[curr_gene, target_gene] = ratio
    pair_values, pair_names = return_lower_triangle(ratio_mat)
    return (pair_values, pair_names)

def pair_anticorrelation_test(data, sorted_by_pair_ratio):
    true_rho = []
    null_rho_dist = []
    p_dist = []
    for i in sorted_by_pair_ratio:
        gene1, gene2 = i
        subset = np.logical_or(data[gene1] > 0, data[gene2] > 0)
        stats = sp.stats.spearmanr(data.loc[subset, gene1], data.loc[subset, gene2])
        shuffle_times = 100
        null_rho = []
        s_gene1 = data.loc[subset, gene1].copy()
        s_gene2 = data.loc[subset, gene2].copy()
        for i in range(shuffle_times):
            np.random.shuffle(s_gene1.values)
            np.random.shuffle(s_gene2.values)
            s_stats = sp.stats.spearmanr(s_gene1, s_gene2)[0]
            null_rho.append(s_stats)
        null_rho = np.array(null_rho)
        true_rho.append(stats[0])
        null_rho_dist.append(null_rho)
        p = sp.stats.mannwhitneyu([stats[0]], null_rho)[1]
        p_dist.append(p)
    null_rho_dist = np.array(null_rho_dist)
    true_rho = np.array(true_rho)
    p_dist = np.array(p_dist)
    sort_id = np.argsort(p_dist)
    null_rho_dist = null_rho_dist[sort_id]
    true_rho = true_rho[sort_id]
    p_dist = p_dist[sort_id]
    pair_names = sorted_by_pair_ratio[sort_id]
    remove = [True if x[0] != x[1] else False for x in pair_names]
    pair_names = pair_names[remove]
    null_rho_dist = null_rho_dist[remove]
    true_rho = true_rho[remove]
    p_dist = p_dist[remove]
    return(pair_names, null_rho_dist, true_rho, p_dist)

def generate_subsample(df_input, sum_reads_input, fractions=[]):
    if np.shape(fractions)[0] == 0:
        fractions = np.logspace(-4, 0, 20)
    weights = (df_input / sum_reads_input)
    length_of_features = np.arange(np.shape(df_input)[0])
    y_output = np.zeros((np.shape(fractions)[0], np.shape(df_input)[1]))
    for fraction, index in zip(fractions, range(np.shape(fractions)[0])):
        subsampled_matrix = np.zeros(np.shape(df_input)).reshape(np.shape(df_input)[0], -1)
        for i in np.arange(np.shape(df_input)[1]):
            ngood = ((weights.iloc[:,i].values)*sum_reads_input[i]*fraction).astype(int)
            nbad = ((sum_reads_input[i]*fraction) - ngood).astype(int)
            random_sample = np.random.hypergeometric(ngood, nbad, int(sum_reads_input[i]*fraction))
            subsampled_matrix[:,i] = random_sample
        y_output[index, :] = count_on_genes(subsampled_matrix, threshold = 0)
    return (fractions[::-1], y_output[::-1])

def calculate_var_mean(data, genes, ignore_zero_values = True, threshold = 0):
    if ignore_zero_values == True:
        var, mean = data[genes].var(axis=0).values, \
                    (data[genes][data[genes] > threshold]).mean(axis=0).values
    else:
        var, mean = data[genes].var(axis=0).values, \
                    data[genes].mean(axis=0).values
    return var, mean

def mat_corr(df1, df2):
    n = len(df1)
    v1, v2 = df1.values, df2.values
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                        df2.columns, df1.columns)

def random_forest_cell_cycle_toxo(test_adata, rh_adata_loc):
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import scanpy.api as sc
    tmp = test_adata.copy()
    tmp.var_names = [x.replace('TGGT1_', '').replace('TGME49_', '') for x in
                     tmp.var_names]
    rh_adata = sc.read_h5ad(rh_adata_loc)
    rh_adata.var_names = [x.replace('TGGT1_', '') for x in rh_adata.var_names]
    cell_cycle_genes = [x.replace('TGGT1_', '') for x in
                        rh_adata.var_names[rh_adata.var[
                            'sam_high_ranked'].values]]
    intersect_genes = np.intersect1d(cell_cycle_genes, tmp.var_names)
    labels = rh_adata.obs['cell_cycle'].values.astype(str)
    X = rh_adata[:, intersect_genes].X.A
    training_data = pd.DataFrame(sklearn.preprocessing.Normalizer().fit_transform(np.log2(X+1)),\
                                 index=rh_adata.obs_names, columns=intersect_genes)
    X = tmp[:, intersect_genes].X.A
    predict_data = pd.DataFrame(sklearn.preprocessing.Normalizer().fit_transform(np.log2(X+1)),\
                                index=tmp.obs_names, columns=intersect_genes)
# Split the data into training and testing sets
    split_fraction = 0.4
    train_features, test_features, train_labels, test_labels =\
    train_test_split(training_data.values, labels, test_size = split_fraction,
                     random_state = 0)
#### Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
#### Train the model on the PCA transform of the training data
    rf.fit(train_features, train_labels)
#### Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
##### Print out the mean absolute error (mae)
    print('Test error: %s' %(np.sum(predictions != test_labels)
                             /np.shape(test_labels)[0]))
    accuracy = 100*(1-(np.sum(predictions != test_labels)
                       /np.shape(test_labels)[0]))
    print('Test accuracy: %s' %accuracy)
#### Use the forest's predict method on the test data
    predicted_labels = rf.predict(predict_data)
    return predicted_labels

def calculate_neighborhood_variation(X, nnm, mode='variance'):
    '''
    X: Expression matrix (generally binarized) where rows are cells and columns
    are genes. Should have the same row length N as 'nnm' adjacency matrix
    N x N.
    nnm: N x N adjacency matrix.
    '''
    size_array = np.arange(X.shape[0])
    if sp.sparse.issparse(X):
        subset = (X > 0).A
    else:
        subset = (X > 0)
    mu = np.zeros(X.shape[1])
    var = np.zeros(X.shape[1])
    for ind, (u, sub) in enumerate(zip(X.T.A, subset.T)):
        check = size_array[sub]
        values = []
        for j in nnm[check]:
            values.append(var_func(u[j], mode=mode))
        mu[ind] = np.mean(values)
        var[ind] = np.var(values)
    return (mu, var)

def var_func(inval, mode='variance'):
    if mode == 'variance':
        return np.var(inval)
    elif mode == 'entropy':
        size = np.shape(inval)[0]
        ones = inval.sum()
        zeros = size - ones
        return sp.stats.entropy([ones/size, zeros/size])
    elif mode == 'residual':
        return np.sum(1 - inval)

def enrich_genes(exp, C, groupby='specific', method='wilcoxon'):
    if groupby=='specific':
        UC = np.unique(C)
        avg = sat.average_by_clusters(exp, C, return_clusters=True)
        UC_array = np.repeat(UC, avg.shape[1]).reshape(avg.shape[0], -1)
        id_index = avg.index.values
        length = len(UC)
        max_exp_cluster = np.array([UC[x] for x in np.argmax(avg.values, axis=0)])
        max2_exp_cluster = np.array([UC[x] for x in
                                     np.argsort(avg.values, axis=0)[-2]])
        E = exp.values
        outvals = []
        for eid, g in enumerate(exp.columns.values):
            sub1 = max_exp_cluster[eid]
            sub2 = max2_exp_cluster[eid]
            v1 = E[C == sub1, eid]
            v2 = E[C == sub2, eid]
            if method == 'wilcoxon':
                _stats, p_val = sp.stats.ranksums(v1, v2)
            elif method == 'ttest':
                _stats, p_val = sp.stats.ttest_ind(v1, v2)
            elif method == 'mannwhitneyu':
                try:
                    _stats, p_val = sp.stats.mannwhitneyu(v1, v2)
                except:
                    _stats = 0
                    p_val = 1
            outvals.append([v1.mean()/v2.mean(), p_val, sub1, sub2])
        out = pd.DataFrame(outvals, index=exp.columns, columns=['fold', 'pval',
                                                                'max1', 'max2'])
        out = out.fillna(0)
    else:
        UC = np.unique(C)
        avg = sat.average_by_clusters(exp, C, return_clusters=True)
        UC_array = np.repeat(UC, avg.shape[1]).reshape(avg.shape[0], -1)
        id_index = avg.index.values
        length = len(UC)
        max_exp_cluster = np.array([UC[x] for x in np.argmax(avg.values, axis=0)])
        E = exp.values
        outvals = []
        for eid, g in enumerate(exp.columns.values):
            sub1 = max_exp_cluster[eid]
            v1 = E[C == sub1, eid]
            v2 = E[C != sub1, eid]
            _stats, p_val = sp.stats.ranksums(v1, v2)
            if method == 'wilcoxon':
                _stats, p_val = sp.stats.ranksums(v1, v2)
            elif method == 'ttest':
                _stats, p_val = sp.stats.ttest_ind(v1, v2)
            elif method == 'mannwhitneyu':
                try:
                    _stats, p_val = sp.stats.mannwhitneyu(v1, v2)
                except:
                    _stats = 0
                    p_val = 1
            outvals.append([v1.mean()/v2.mean(), p_val, sub1])
        out = pd.DataFrame(outvals, index=exp.columns, columns=['fold', 'pval',
                                                                'max1'])
        out = out.fillna(0)
    return out
