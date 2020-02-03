import pandas as pd
import os
import numpy as np
import sc_utilities as ut
import plot
import graph
import statsmodels.nonparametric.api as smnp
import pickle, gzip
import matplotlib.pyplot as plt
import pseudotime
import analysis
scriptDirectory = os.path.dirname(os.path.realpath(__file__))

def loom_to_h5ad(loom_in, fix_cell_names=True):
    try:
        import anndata
        out_h5ad = anndata.AnnData(loom_in.X.copy())
        out_h5ad = anndata.AnnData(loom_in.X.copy())
        out_h5ad.obs = loom_in.obs.copy()
        if fix_cell_names == True:
            cell_names = [x.split(':')[1].replace('.bam', '') for x \
                          in loom_in.obs_names]
            nmn = []
            for c in cell_names:
                b = c.split('_')
                if len(b) > 3:
                    c = '_'.join(b[:3])
                nmn.append(c)
            out_h5ad.obs.index = nmn
        out_h5ad.var = loom_in.var.copy()
        for key in loom_in.layers.keys():
            out_h5ad.layers[key] = loom_in.layers[key].A.copy()
        return out_h5ad
    except ImportError:
        print('Missing anndata')


def linreg_fit(data, truth, fit_intercept=True):
    from sklearn.linear_model import LinearRegression
    if isinstance(data, pd.DataFrame):
        X = data.mean(axis=0).values.reshape(-1, 1)
        y = truth.values
    else:
        X = np.mean(data, axis=0).reshape(-1, 1)
        y = np.array(truth)
    reg = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    m = reg.coef_[0]
    b = reg.intercept_
    r2 = reg.score(X, y)
    return (X, y, (m, b, r2))


def quantify_transcripts(data,
                         log2=False,
                         transpose=False,
                         plot=False,
                         fit_intercept=True,
                         pseudocount=1,
                         detectionLimit=None,
                         filename='',
                         verbose=False,
                         logx=True,
                         s=40,
                         figsize=(3, 3),
                         absolute=True,
                         return_fit=False,
                         mode='linfit',
                         min_threshold=2,
                         **args):
    '''
    We calculate the number of transcripts based on a linear regression fit of
    ERCC in the samples. We assume data is normalized (e.g. CPM or TPM), it
    may or may not be log transformed. It calculates sensitivity of measurement
    (50% detection limit) first, then uses linear regression to quantify
    abundnace of transcripts above the detection limit.
    There are modes available:
        "linfit" : is to fit with linear regression in order to calculate the
        slope and intercept of observed values (e.g. CPM) to the absolute
        amount of ERCC. This mode takes log2 or linear normalized values (CPM)
        as input.
        "em" : is expectation maximization. In essence, this just calculates
        the fraction of reads that map to ERCC and calculate the expected
        amount based on total ERCC spike-in. This mode takes raw reads as
        input.
    '''

    from sklearn.linear_model import LinearRegression
    if absolute == True:
        ercc_amount = get_ercc_amount(**args)
    else:
        ercc_genes = [x for x in data.columns.values if 'ERCC-' in x]
        ercc_amount = data[ercc_genes].copy()
        ercc_amount = ercc_amount.mean(axis=0)
    if transpose == True:
        data = data.copy().T
    else:
        data = data.copy()
    if not detectionLimit:
        if verbose == True:
            print('detectionLimit not provided, calculating sensitivity of\n\
                  measurement...')
        clf, detectionLimit = quantify_sensitivity(ercc_amount, data)
        if verbose == True:
            print('Detection limit determined: %s' %
                  np.round(detectionLimit, 2))
    if log2 == True:
        data = np.log2(data + pseudocount)
    ercc_amount = np.log2(ercc_amount[ercc_amount > detectionLimit])
    ercc_genes = np.intersect1d(ercc_amount.index.values, data.columns.values)
    if mode == 'linfit':
        X, y, (m, b, r2) = linreg_fit(data[ercc_genes],
                                      ercc_amount.loc[ercc_genes],
                                      fit_intercept=fit_intercept)
        fitted_output = data * m + b
        fitted_output[fitted_output < 0] = 0
        fitted_output[data < 0] = 0
        if plot == True or return_fit == True:
            test_x = np.linspace(np.min(X), np.max(X), 50).reshape(-1, 1)
            pred_y = test_x * m + b
        if plot == True:
            fig, ax = plt.subplots(figsize=figsize)
            if logx == False:
                y = 2**y
                plot_fitted_output = 2**(fitted_output[ercc_genes].copy()\
                                         .mean(axis=0)).values
                pred_y = 2**pred_y
                xlabel = 'Expected amount'
            else:
                plot_fitted_output = fitted_output[ercc_genes].copy()\
                        .mean(axis=0).values
                xlabel = (r'Expected amount (log$_2$)')
            ax.scatter(y,
                       X.flatten(),
                       label='Ground truth',
                       c='C0',
                       s=s,
                       lw=1,
                       zorder=10)
            ax.plot(pred_y,
                    test_x.flatten(),
                    label='Linear fit',
                    c='r',
                    linestyle='--')
            combined_y = np.concatenate([pred_y.flatten(), y.flatten()])
            ymin, ymax = np.round(combined_y.min(), 0) - 1,\
                            np.round(combined_y.max(), 0) + 1
            if logx == True:
                ax.set_xticks(np.arange(ymin, ymax, 1))
            else:
                ax.set_xticks([-2, 0, 2, 4, 6, 8])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'CPM (log$_2$)')
            ax.text(0.05,
                    0.7,
                    r'R$^2$: %s' % np.round(r2, 2),
                    transform=ax.transAxes,
                    zorder=10)
            ax.set_xlim(-2.5, 9)
            ax.legend(bbox_to_anchor=(1, 0.9), frameon=False)
            ax.set_title('Measurement accuracy', fontsize=12)
            if filename != '':
                fig.savefig(filename, bbox_inches='tight', transparent=True)
        fitted_output = 2**fitted_output
        fitted_output[fitted_output < detectionLimit] = 0
        if return_fit == False:
            return fitted_output
        else:
            return fitted_output, (pred_y, test_x.flatten(), r2)
    elif mode == 'em':
        data[data < min_threshold] = 0
        ercc_amount = get_ercc_amount()
        nominator = (data).values
        denominator = (data.values).sum(axis=1)
        ercc_fraction = (data[ercc_amount.index]).sum(axis=1) / denominator
        fraction = np.divide((data).T, denominator).T
        total_mrna = ercc_amount.sum() / ercc_fraction
        fitted_output = pd.DataFrame(np.multiply(fraction.T, total_mrna).T,
                                       index=data.index,
                                       columns=data.columns)
        return fitted_output
    else:
        raise ValueError('mode must be either "em" or "linfit"')


def quantify_sensitivity(ercc_data,
                         expression_mat,
                         log2=False,
                         min_expression=0,
                         minimum_ercc_amount=-4,
                         plot=False,
                         figsize=(3, 3),
                         filename='',
                         minimal_threshold=0.5,
                         **args):
    from sklearn.linear_model import LogisticRegression
    '''
    Refer to Svensson Valentine et al. (2017, Nature Methods) for
    quantification of scRNA-seq detection sensitivity using logistic regression

    Parameters:
    -----------
    ercc_data: A panda series with index being ERCC gene names,
                contains absolute amount of ERCC spike-in based on dilution
                calculation.
    expression_mat: Must contain ERCC gene names as in ercc_data index.
                Data is expected to be in log2 space.

    Returns:
    -------
    logisticRegr: The logistic regression model from sklearn.
    half_max: The mid-point of logistic regression in linear space.
              This should correspond to detection limit of ERCC in copy number.
    '''

    if log2 == True:
        logged_molecule_counts = np.log2(ercc_data.copy())
    else:
        logged_molecule_counts = (ercc_data.copy())
    expression_data = (expression_mat.copy())
    logged_molecule_counts = logged_molecule_counts[
        logged_molecule_counts > minimum_ercc_amount]
    unique_vals, ind = np.unique(logged_molecule_counts.values,
                                 return_inverse=True)
    ercc_genes = logged_molecule_counts.index.values
    ## We store a giant vector of true ERCC amount
    X = np.array([])
    ## A vector to store bernoulli variable of observation
    ## Either 1 for detected or 0 for not detected
    bernoulli = np.array([])
    for u in unique_vals:
        ## Using the ERCC which have the same number of ground truth
        ## number of molecules, we retrieve the
        ## expression data and flatten it.
        expression_counts = expression_data[ercc_genes[
            logged_molecule_counts.values == u]]
        ## We create a new variable
        detections = (expression_counts >
                      min_expression).astype(int).values.flatten()
        X = np.concatenate([X, np.repeat(u, np.shape(detections)[0])])
        bernoulli = np.concatenate([bernoulli, detections])
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X.reshape(-1, 1), bernoulli.reshape(-1, 1))
    test_x = np.linspace(np.min(X), np.max(X), 10**5)
    predicted_y = logisticRegr.predict((test_x).reshape(-1, 1))
    predicted_1_molecule = (logistic_model(np.array([1]), logisticRegr.coef_,
                        logisticRegr.intercept_).ravel())
    plot_y = logistic_model(test_x, logisticRegr.coef_,
                            logisticRegr.intercept_).ravel()
    min_difference_id = np.argmin(np.abs(plot_y - minimal_threshold))
    half_max = (test_x[min_difference_id])
    if plot == True:
        fig, ax = plt.subplots(figsize=figsize)
        for u in np.unique(X):
            y_values = bernoulli[X == u]
            on_alpha = y_values.sum() / np.shape(y_values)[0]
            off_alpha = 1 - on_alpha
            ax.scatter((u), 1, alpha=on_alpha, c='C0', **args)
            ax.scatter((u), 0, alpha=off_alpha, c='C0', **args)
        ax.plot(test_x,
                plot_y,
                ls='-',
                c='green',
                label='Logistic regression fit')
        ax.set_xlabel(r'Expected amount (log$_{2}$)', fontsize=12)
        ax.set_title('Measurement sensitivity', fontsize=12)
        ax.plot([np.min(test_x), (test_x[min_difference_id])],
                np.repeat(minimal_threshold, 2),
                ls='--',
                c='k')
        ax.plot(np.repeat((half_max), 2) , [0, minimal_threshold], ls='--',
                c='k', label='%s detected (linear):\n%s' \
                %(str(np.round(minimal_threshold*100, 2))+'%',\
                  np.round(half_max,1)))
        ax.plot([0, 1] , [0, predicted_1_molecule], ls=':',
                c='k', label='single molecule sensitivity: \n%s' \
                %(str(np.round(predicted_1_molecule[0]*100, 1))+'%'))
        ax.legend(bbox_to_anchor=(1, 0.9), frameon=False)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1])
        ax.set_yticklabels(['0% detected', '10%', '20%',
                            '30%', '50% detection limit',
                            '70%', '80%', '90%', '100% detected'])
        ax.set_xscale('log')
        if filename != '':
            fig.savefig(filename, bbox_inches='tight', transparent=True)
    return ((logisticRegr, half_max))


def get_ercc_amount(ercc_loc='', keep_only=[], which_mix=1, \
                    ercc_volume_added=8, ercc_dilution_factor=600000):
    '''
    Calculate the amount of ercc there is. We do it in log10 space to avoid
    numerical flow issues.

    which_mix: there are two ERCC RNA standard mixes available.
    ercc_dilution_factor: The fold dilution of the ERCC. At Quake lab we
                            typically do 600000 dilution.
    ercc_volume_added: The volume (nanoliters) of ERCC added to the lysis
                        buffer per well / reaction. At Quake lab we typically
                        add 8 nL.
    '''

    if ercc_loc != '':
        ercc_mat = pd.read_csv(ercc_loc, sep='\t', index_col=0)
    else:
        ercc_mat = pd.read_csv(scriptDirectory +
                               '/dataStructure/ERCC_concentration.tsv.gz',
                               sep='\t',
                               index_col=0)
    if np.shape(keep_only)[0] > 0:
        if str(which_mix) == '1':
            ercc_conc = ercc_mat['concentration in Mix 1 (attomoles/ul)'].loc[
                keep_only]
        else:
            ercc_conc = ercc_mat['concentration in Mix 2 (attomoles/ul)'].loc[
                keep_only]
        ercc_amount = 10**(np.log10(ercc_conc) + np.log10(
            ercc_volume_added/1000) + np.log10(1 / ercc_dilution_factor)\
            + np.log10(6.02214076) + 23 - 18)
    else:
        if str(which_mix) == '1':
            ercc_conc = ercc_mat['concentration in Mix 1 (attomoles/ul)']
        else:
            ercc_conc = ercc_mat['concentration in Mix 2 (attomoles/ul)']
        ercc_amount = 10**(np.log10(ercc_conc) + np.log10(
            ercc_volume_added/1000) + np.log10(1 / ercc_dilution_factor)\
            + np.log10(6.02214076) + 23 - 18)
    return ercc_amount


def bin_groups(X, bin_size=None):
    '''
    We separate values in X as chunks of n length

    Parameters:
    -----------

    bin_size - The maximum number of elements in a chunk
    '''

    X = np.nan_to_num(np.array(X))
    if not bin_size:
        bin_size = int(np.shape(X)[0] / 20)
        bins = np.ceil(np.shape(X)[0] / bin_size)
    else:
        bins = np.ceil(np.shape(X)[0] / bin_size)
    order = np.argsort(X)
    bin_edges = np.arange(bins)
    groups = []
    indices = []
    for ind, i in enumerate(bin_edges):
        if ind < (np.shape(bin_edges)[0]):
            min_range = int(ind * bin_size)
            max_range = int((ind + 1) * bin_size)
            subset = order[min_range:max_range]
            indices.append(subset)
            groups.append(X[subset])
    return (groups, indices)


def cpm_normalize(data, transpose=False, median=True, denominator=10**6):
    ndarray = False
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data.copy())
        ndarray = True
    if transpose == True:
        data = data.T
    reads_sum = data.sum(axis=1)
    if median == True:
        normalized_data = data.div(reads_sum, axis=0) * np.median(reads_sum)
    else:
        normalized_data = data.div(reads_sum, axis=0) * denominator
    if ndarray == True:
        normalized_data = np.array(normalized_data)
    return normalized_data


def get_product(gene, dictionary):
    '''
    Takes a single gene ID or a list of gene IDs and return the matching
    product names found in the dictionary
    '''
    output = []
    if isinstance(gene, str):
        gv = np.array([gene])
    else:
        gv = gene.copy()
    gv = np.array(gv)
    for g in gv:
        if g not in dictionary.keys():
            output.append('')
        else:
            output.append(dictionary[g])
    if isinstance(gene, str):
        return (output[0])
    else:
        return (np.array(output))


def get_gene(product, dictionary):
    '''
    Takes a product name and return gene IDs whose protein product contains
    the provided product name as a substring.
    '''
    if type(product) is str:
        output = [[str(x), str(y)]
                  for x, y in zip(dictionary.keys(), dictionary.values())
                  if product in str(y)]
    else:
        output = []
        for i in product:
            output.extend([[str(x), str(y)]
                      for x, y in zip(dictionary.keys(), dictionary.values())
                      if i in str(y)])
    output = pd.DataFrame(output, columns=['gene', 'product'])
    if len(output) >= 1:
        return output
    else:
        print('No matching gene name found')
        return []


def average_by_clusters(mat,
                        clusters,
                        return_clusters=False,
                        ignore_zeros=False):
    unique, counts = np.unique(clusters, return_counts=True)
    return_df = False
    if isinstance(mat, pd.DataFrame):
        col_names = mat.columns.values
        row_names = mat.index.values
        mat = mat.T.values
        return_df = True
    else:
        col_names = np.arange(mat.shape[0])
        row_names = np.arange(mat.shape[1])
    if len(np.shape(mat)) < 2:
        if return_clusters == False:
            new_mat = np.zeros(np.shape(clusters)[0])
            for i in unique:
                new_mat[clusters == i] = calculate_vector_mean(
                    mat[clusters == i], ignore_zeros=ignore_zeros)
        else:
            new_mat = np.zeros(np.shape(unique)[0])
            for count, i in enumerate(unique):
                new_mat[count] = calculate_vector_mean(
                    mat[clusters == i], ignore_zeros=ignore_zeros)

    else:
        if return_clusters == False:
            new_mat = np.zeros(np.shape(mat))
            size_of_mat = new_mat.shape[0]
            for i, cnt in zip(unique, counts):
                new_mat[:, clusters == i] = np.repeat(
                    calculate_vector_mean(mat[:, clusters == i],
                                          ignore_zeros=ignore_zeros,
                                          axis=1),
                    cnt).reshape(size_of_mat, cnt)
        else:
            new_mat = np.zeros((np.shape(mat)[0], np.shape(unique)[0]))
            count = 0
            for i, cnt in zip(unique, counts):
                new_mat[:, count] = calculate_vector_mean(
                    mat[:, clusters == i], axis=1, ignore_zeros=ignore_zeros)
                count += 1
    if return_df == True or return_clusters == True:
        if return_clusters == True:
            new_mat = pd.DataFrame(new_mat.T, index=unique, columns=col_names)
        else:
            new_mat = pd.DataFrame(new_mat.T, \
                                   index=row_names, columns=col_names)
    return new_mat

def detection_by_clusters(mat,
                          clusters,
                          return_clusters=False,
                          min_threshold=0):
    unique, counts = np.unique(clusters, return_counts=True)
    return_df = False
    if isinstance(mat, pd.DataFrame):
        col_names = mat.columns.values
        row_names = mat.index.values
        mat = mat.T.values
        return_df = True
    else:
        col_names = np.arange(mat.shape[0])
        row_names = np.arange(mat.shape[1])
    new_mat = np.zeros((np.shape(mat)[0], np.shape(unique)[0]))
    count = 0
    for i, cnt in zip(unique, counts):
        new_mat[:, count] = (mat[:, clusters == i] >
                             min_threshold).sum(axis=1) / (clusters==i).sum()
        count += 1
    if return_df == True or return_clusters == True:
        new_mat = pd.DataFrame(new_mat.T, index=unique, columns=col_names)
    return new_mat



def calculate_vector_mean(V, ignore_zeros=False, axis=None):
    if ignore_zeros == False:
        return np.mean(V, axis=axis)
    else:
        t = V.copy()
        t[t <= 0] = np.nan
        t = np.nanmean(t, axis=axis)
        t = np.nan_to_num(t)
        return t


def merge_identical_genes(df, product_d):
    new_product_d = {}
    original_gene_names = []
    new_index = []
    new_df = df.copy()
    for gene_id in df.index.values:
        original_gene_names.append(product_d[gene_id])
    original_gene_names = np.array(original_gene_names)
    processed_gene_names = {}
    for i, j in zip(df.index.values, original_gene_names):
        if 'hypothetical_protein' not in j.lower() and j != 'NA':
            ind = np.where(original_gene_names == j)[0]
            if np.shape(ind)[0] > 1:
                if j not in processed_gene_names.keys():
                    new_i = ':'.join(df.index[ind])
                    new_product_d[new_i] = j
                    new_index.append(i)
                    processed_gene_names[j] = new_i
                else:
                    new_index.append(processed_gene_names[j])
            else:
                new_product_d[i] = j
                new_index.append(i)
        else:
            new_product_d[i] = j
            new_index.append(i)
    new_df.index = new_index
    new_df = new_df.groupby(new_df.index).sum(axis=1)
    return new_df, new_product_d


def hill_func(x, k, h):
    y = 1 / (k + x**h)
    return y


def logistic_func(x, k, h):
    y = k / (1 + h * np.exp(-x))
    return y


def logistic_model(x, m, b):
    return (1 / (1 + np.exp(-(x * m + b))))


def linear_model(x, m, b):
    return (m * x + b)


def mat_corr(df1, df2):
    n = len(df1)
    if isinstance(df1, pd.DataFrame) or isinstance(df2, pd.DataFrame):
        v1, v2 = df1.values, df2.values
        sums = np.multiply.outer(v2.sum(0), v1.sum(0))
        stds = np.multiply.outer(v2.std(0), v1.std(0))
        return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n, df2.columns,
                            df1.columns)
    else:
        v1, v2 = df1, df2
        sums = np.multiply.outer(v2.sum(0), v1.sum(0))
        stds = np.multiply.outer(v2.std(0), v1.std(0))
        return ((v2.T.dot(v1) - sums / n) / stds / n)


def sg_filter(y, window_size, order, deriv=0, rate=1):
    '''
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only
                                                smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    '''
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def det_cpm(ad, genes, threshold=None, by='absolute', nonzero_mean=False):
    '''
    Calculate the detection probability and mean expression values from AnnData
    object.
    '''
    if by == 'absolute':
        obs = pd.DataFrame(ad[:, genes].layers['absolute'].A)
    else:
        obs = pd.DataFrame(ad[:, genes].X.A)
    det = obs.copy()
    if not threshold and by == 'absolute':
        dl = ad.uns['detection_limit']
    elif not threshold and by != 'absolute':
        dl = 0
    else:
        dl = threshold
    det[det <= dl] = 0
    det[det > 0] = 1
    det = det.sum(axis=0).values / (det.shape[0])
    if nonzero_mean == True:
        std = obs[obs > 0].std(axis=0).fillna(0).values
        obs = obs[obs > 0].mean(axis=0).fillna(0).values
    else:
        std = obs.std(axis=0).fillna(0).values
        obs = obs.mean(axis=0).fillna(0).values
    return (obs, det, std)


def make_paired_matrix_from_vec(cols, ind, sep='-'):
    """
    Takes two vectors containing strings, cols and ind, and create a matrix by
    joining the elements from both vector, col-wise.
    """
    data = {j: [c+sep+i for i in ind] for
            j, c in enumerate(cols)}
    return pd.DataFrame(data).values


def vec_translate(a, my_dict):
    """
    Translate all elements in a vector / matrix according to a dictionary
    mapping scheme
    """
    return np.vectorize(my_dict.__getitem__)(a)

def shuffle(X):
    """
    Shuffle an input matrix
    """
    shape = np.shape(X)
    if isinstance(X, pd.DataFrame):
        is_df = True
        shuffled = X.copy().values.flatten()
    else:
        is_df = False
        shuffled = X.copy().flatten()
    np.random.shuffle(shuffled)
    shuffled = shuffled.reshape(shape)
    if is_df is True:
        shuffled = pd.DataFrame(shuffled, index=X.index, columns=X.columns)
    return shuffled

def stratify(X, labels, out_type=dict):
    D = np.array(X)
    L = np.array(labels)
    assert len(D) == len(L), 'the length of "X" must match that of "labels"'
    unique_labels = np.unique(labels)
    if out_type is dict:
        out = {}
        for u in unique_labels:
            out[u] = X[L == u]
    else:
        out = []
        for u in unique_labels:
            out.append(X[L == u])
    return out
