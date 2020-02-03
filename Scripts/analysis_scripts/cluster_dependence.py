from _loadlib.toxo_lib import *
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
script_dir = (os.path.dirname(os.path.realpath(__file__)))

def shuffle_data(X):
    if isinstance(X, pd.DataFrame):
        shuffled_x = X.values.copy()
    else:
        shuffled_x = X.copy()
    np.random.shuffle(shuffled_x)
    return shuffled_x

def compute_null_distributions(X, cluster_labels, num=100, verbose=True):
    null_v = []
    if isinstance(X, pd.DataFrame):
        X = X.values.copy()
    unique_labels = np.unique(cluster_labels)
    for i in range(num):
        if verbose == True and i%10 == 0:
            print('Currently on iteration %s' %(i+1))
        each_iteration = []
        for ind, observed_gene in enumerate(X.T):
            each_gene = []
            shuffled = shuffle_data(observed_gene)
            for u in unique_labels:
                each_gene.append(sp.stats.ranksums(observed_gene[cluster_labels == u],\
                                                   shuffled[cluster_labels==u])[0])
            each_iteration.append(each_gene)
        null_v.append(each_iteration)
    null_v = np.array(null_v)
    return null_v

def automatic_supervariant_genes_finder(X, projection, genes, negativeGenes,\
                        shuffle_times = 100, cluster_size=80, verbose=True,
                        method='default'):
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=cluster_size).fit_predict(projection)
    if method == 'default':
        null_output = compute_null_distributions(X, labels, num=shuffle_times,
                                    verbose=verbose)
        null_output = compute_null_distributions_improved(X, labels,
                                              num=shuffle_times, verbose=verbose)
        df_null_output = pd.DataFrame(null_output.max(axis=2),
                              index=np.arange(shuffle_times), columns = X.columns)
        mean, std = df_null_output[negativeGenes].mean(axis=0).mean(), \
                    df_null_output[negativeGenes].mean(axis=0).std()
        zscores = np.abs((df_null_output.mean(axis=0) - mean) / std)
        negative_pvals = pd.DataFrame(sp.stats.norm.sf(zscores),
                                      index = zscores.index, columns=['pvals'])
        negative_pvals['zscores'] = zscores.values
    elif method == 'test':
        negative_pvals = compute_kde_null(X, num=shuffle_times,
                                          verbose=verbose)
    negative_pvals = negative_pvals.sort_values('pvals', ascending=False)
    genes_pvals = negative_pvals.loc[genes].copy()
    genes_pvals = genes_pvals.sort_values('pvals', ascending=False)
    negative_pvals = negative_pvals.loc[[x for x in
                             negative_pvals.index.values if x not in genes]]
    return genes_pvals, negative_pvals

def get_dependence(D, projection, negativeGenes, gene_to_product,
                   shuffle_times=100, cluster_size=80):
    norm_data = np.log2(D.copy()+1)
    norm_data = norm_data.div(norm_data.max(axis=0), axis=1)
    df_pvals, ercc_pvals = \
    automatic_supervariant_genes_finder(norm_data, projection=projection,\
                        genes=np.setdiff1d(D.columns.values, negativeGenes), \
            negativeGenes=negativeGenes, shuffle_times=shuffle_times,
                                        cluster_size=cluster_size)
    df_pvals['product'] = [x for x in sat.get_product(\
                                [x for x in df_pvals.index.values],\
                                              gene_to_product)]
    ercc_pvals['product'] =  [x for x in sat.get_product(\
                                [x for x in ercc_pvals.index.values],\
                                              gene_to_product)]
    return df_pvals, ercc_pvals

def get_development_cc_violin(df_dot, all_genes_dict, variant_genes,
                              families=None):
    final = pd.DataFrame()
    if not families:
        families = all_genes_dict.keys()
    for family in families:
        genes = np.intersect1d(variant_genes, all_genes_dict[family])
        X = df_dot.loc[genes].copy()
        length = np.shape(X)[0]
        if df_dot.shape[1] > 1:
            D = pd.DataFrame(0, index=range(length*2),
                             columns=['Score', 'Process', 'Organelle sets'])
            D['Process'] = np.concatenate([np.repeat(X.columns.values[0], length),
                                          np.repeat(X.columns.values[1], length)])
            D['Score'] = np.concatenate([X['Development'].values,
                                        X['Cell cycle'].values])
        else:
            D = pd.DataFrame(0, index=range(length*2),
                             columns=['Score', 'Process', 'Organelle sets'])
            D['Process'] = np.concatenate([np.repeat('Development', length),
                                      np.repeat(X.columns.values[0], length)])
            D['Score'] = np.concatenate([np.repeat(np.nan, length),
                                        X['Cell cycle'].values])

        D['Organelle sets'] = family
        final = pd.concat([final, D], axis=0)
    return final

def scramble(a, axis=0):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

def get_gene_family(gpd, prefix='TGME49_'):
    ribosomalRNA = sat.get_gene('ribosomal RNA', gpd)['gene'].values
    ribosomalRNA = np.array([x for x, y in zip(ribosomalRNA,
                       sat.get_product(ribosomalRNA, gpd))
                             if 'ribosomal RNA ' not in y])
    gene_dict = {}
    gene_dict['Ribosomal RNA'] = ribosomalRNA
    genes_of_interest = ['SRS', 'ERCC-', 'rhoptry', 'ROP', 'RON', 'microneme',
                         'IMC', 'hypothetical protein', 'gondii family A',
                        'gondii family B', 'gondii family C', 'dense granule',
                        'GRA']
    key_names = ['SRS', 'ERCC', 'ROP', 'ROP', 'ROP', 'MIC', 'IMC',
                 'Hypothetical', 'Family A', 'Family B', 'Family C',
                 'GRA', 'GRA']
    gra = [prefix + str(x) for x in pd.read_csv(script_dir+'/Bradley_GRAs.csv',
                                           sep=',', index_col=0).index.values]
    gene_dict['GRA'] = gra
    seen = []
    for ind, i in enumerate(genes_of_interest):
        genes = sat.get_gene(i, gpd)['gene'].values.astype(str)
        seen = np.concatenate([seen, genes])
        if key_names[ind] not in list(gene_dict.keys()):
            gene_dict[key_names[ind]] = genes
        else:
            gene_dict[key_names[ind]] = np.unique(np.concatenate([
                gene_dict[key_names[ind]], genes]))
    seen = np.array(seen).flatten()
    all_g = np.array(list(gpd.keys()))
    gene_dict['Others'] = np.setdiff1d(all_g, seen)
    return gene_dict

class knn_approach():
    def __init__(self, X, weights, bw=None, num=100, k=5, metric='euclidean',
                 verbose=True):
        if isinstance(X, pd.DataFrame):
            self.X = X.values.copy()
        else:
            self.X = X.copy()
        if isinstance(weights, pd.DataFrame):
            self.geneNames = weights.columns.values
            self.weights = weights.values.copy()
        else:
            self.geneNames = []
            self.weights = weights.copy()
        self.Xpi = X*np.pi/180
        self.metric = metric
        self.num = num
        self.verbose = verbose
        self.k = k

    def run(self, approach='knn', verbose=True):
        self.verbose = verbose
        if approach == 'knn':
            dist = sat.ut.compute_distances(self.X, self.metric)
            adj = sat.ut.dist_to_nn(dist, self.k)
        self.scores = np.zeros((self.num, self.weights.shape[1]))
        for i in range(self.num):
            shuffled = scramble(self.weights.copy(), axis=0)
            if verbose == True and i%10 == 0:
                print('Currently on iteration %s' %(i+1))
            if approach == 'knn':
                obs_nn_avg = (adj/self.k).dot(self.weights)
                obs_nn_avg = obs_nn_avg/obs_nn_avg.sum(axis=0)*\
                                self.weights.sum(axis=0)
                sh_nn_avg = (adj/self.k).dot(shuffled)
                sh_nn_avg = sh_nn_avg/sh_nn_avg.sum(axis=0)*self.weights.sum(axis=0)
                for j, (obs, nn_avg, sh) in enumerate(zip(self.weights.T,
                                                  obs_nn_avg.T, sh_nn_avg.T)):
                    self.scores[i, j] = ks_2samp(nn_avg, sh)[1]
        if approach == 'knn':
            self.nn_avg = obs_nn_avg
            self.nn_avg_sh = sh_nn_avg
            self.log_scores = -np.log10(self.scores).T
            if len(self.geneNames) > 0:
                self.df_scores = pd.DataFrame(self.log_scores.mean(axis=1),
                                     index=self.geneNames, columns=['pvals'])
            else:
                self.df_scores = pd.DataFrame(self.log_scores.mean(axis=1),
                                     index=range(self.weights.shape[1]),
                                              columns=['pvals'])
            self.df_scores = np.sqrt(self.df_scores.sort_values('pvals'))
