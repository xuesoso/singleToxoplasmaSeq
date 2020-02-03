from _loadlib.toxo_lib import *
script_dir = os.path.dirname(os.path.abspath( __file__ ))

def get_nonzero_df(adata_in, gene_list, ercc_conc=[], min_threshold=0,
                   gene_dictionary=None):
    out = pd.DataFrame(0, index=gene_list, columns=['nonzero mean', 'mean',
                                                    'fraction_on', 'ercc',
                                                    'averaged'])
    ## We try to find the mode of non-zero mean expression
    vals = pd.DataFrame(adata_in[:, gene_list].layers['absolute'].toarray())
    out.loc[gene_list, 'mean'] = vals.mean(axis=0).values
    out.loc[gene_list, 'nonzero mean'] = vals[vals > min_threshold].mean(axis=0).values
    out.loc[gene_list, 'fraction_on'] = (np.greater(vals, min_threshold).sum(axis=0) / np.shape(vals)[0]).values
    if len(ercc_conc) > 0:
        out['ercc'] = ercc_conc.loc[gene_list].values
        for i in (np.unique(out['ercc'].values)):
            out.loc[(out['ercc'] == i).values, 'averaged'] =\
            out.loc[(out['ercc'] == i).values, 'nonzero mean'].mean()
    if gene_dictionary:
        out['product'] = [gene_dictionary[x] for x in out.index.values]
    out = out.sort_values('fraction_on', ascending=False)
    return out

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

def fig4_variant_genes(ad, min_pval=0.05, method='detection', basis='absolute',
                      threshold=None, nonzero_mean=False):
    gpd = {x:y for x, y in zip(ad.var.index.values, ad.var['product'].values)}
    all_genes = ad.var_names.values
    rop = sat.get_gene('ROP', gpd)['gene'].values
    ercc = sat.get_gene('ERCC-', gpd)['gene'].values
    srs = sat.get_gene('SRS', gpd)['gene'].values
    gra = sat.get_gene('GRA', gpd)['gene'].values
    obs_d, det_d, std_d = {}, {}, {}
    obs, det, std = sat.det_cpm(ad, all_genes, by=basis,
                                nonzero_mean=nonzero_mean)
    obs_d['All'], det_d['All'], std_d['All'] = obs, det, std
    obs, det, std = sat.det_cpm(ad, ercc, by=basis, threshold=threshold,
                                nonzero_mean=nonzero_mean)
    obs_d['ERCC'], det_d['ERCC'], std_d['ERCC'] = obs, det, std
    obs, det, std = sat.det_cpm(ad, rop, by=basis, threshold=threshold,
                                nonzero_mean=nonzero_mean)
    obs_d['ROP'], det_d['ROP'], std_d['ROP'] = obs, det, std
    obs, det, std = sat.det_cpm(ad, gra, by=basis, threshold=threshold,
                                nonzero_mean=nonzero_mean)
    obs_d['GRA'], det_d['GRA'], std_d['GRA'] = obs, det, std
    obs, det, std = sat.det_cpm(ad, srs, by=basis, threshold=threshold,
                                nonzero_mean=nonzero_mean)
    obs_d['SRS'], det_d['SRS'], std_d['SRS'] = obs, det, std
    test_x = np.linspace(np.min(obs_d['All']), np.max(obs_d['All']), 1000)
    if method == 'detection':
        from statsmodels.stats import weightstats as stests
        popt, pcov = sp.optimize.curve_fit(sat.logistic_model,
                                           np.log10(obs_d['ERCC']+0.5),
                                           det_d['ERCC'])
        pred_y = sat.logistic_model(np.log10(test_x+0.5), *popt)
        order = np.argsort(test_x)
        test_x = test_x[order]
        pred_y = pred_y[order]
        real_pred = sat.logistic_model(np.log10(obs_d['All']+0.5), *popt)
        diff = pd.DataFrame(real_pred-det_d['All'], index=all_genes)
        non_ercc = np.setdiff1d(diff.index.values, ercc)
        mean_diff, std_diff = diff.loc[non_ercc].mean().values,\
                diff.loc[non_ercc].std().values
        pvals = []
        z = (diff.loc[non_ercc] - mean_diff) / std_diff
        for i in z:
            pvals.append(1 - sp.stats.norm.cdf(z))
        pvals = np.array(pvals).flatten()
        variant_genes = non_ercc[np.logical_and(pvals < min_pval,
                                                diff.loc[non_ercc].\
                                                values.flatten() > 0)]
        subset = [True if x in variant_genes else False for x in all_genes]
        obs_d['Variant'], det_d['Variant'] = obs_d['All'][subset],\
                                                det_d['All'][subset]
        return obs_d, det_d, test_x, pred_y, variant_genes, diff
    elif method == 'cv':
        from sklearn.linear_model import LinearRegression
        cv_d = {}
        for k in obs_d.keys():
            val = np.nan_to_num(std_d[k] / obs_d[k])
            cv_d[k] = val
        X = (obs_d['ERCC'])
        nonzero = X > 0
        X = X[nonzero].reshape(-1, 1)
        y = (cv_d['ERCC'])
        y = y[nonzero]
        reg = LinearRegression().fit(np.log10(X), np.log10(y))
        test_x = np.linspace(np.min(obs_d['All'][obs_d['All'] > 0]),
                             np.max(obs_d['All']), 1000)
        pred_y = reg.predict(np.log10(test_x).reshape(-1, 1))
        order = np.argsort(test_x)
        test_x = test_x[order]
        pred_y = pred_y[order]
        nonzero_X = obs_d['All'] > 0
        real_pred = np.repeat(0, len(obs_d['All']))
        real_pred[nonzero_X] = reg.predict(np.log10(
                                obs_d['All'][nonzero_X]).reshape(-1, 1))
        real_pred[nonzero_X == False] = 0
        diff = pd.DataFrame(real_pred-np.nan_to_num(np.log10(cv_d['All'])),
                            index=all_genes)
        pvals = []
        non_ercc = np.setdiff1d(diff.index.values, ercc)
        for i in diff.loc[non_ercc].values:
            pvals.append(sp.stats.ranksums(i, diff.loc[ercc].values.flatten())[1])
        pvals = np.array(pvals)
        variant_genes = non_ercc[np.logical_and(pvals < min_pval,
                                                diff.loc[non_ercc].\
                                                values.flatten() > 0)]
        subset = [True if x in variant_genes else False for x in all_genes]
        obs_d['Variant'], cv_d['Variant'] = obs_d['All'][subset],\
                                                cv_d['All'][subset]
        return obs_d, cv_d, test_x, pred_y, variant_genes, diff
