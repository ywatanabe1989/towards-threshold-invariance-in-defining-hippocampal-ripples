from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
import numpy as np

def multicompair(data, labels, testfunc=None):
    # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
    _labels = labels.copy()
    # Set up the data for comparison (creates a specialised object)
    for i_labels in range(len(_labels)):
        _labels[i_labels] = [_labels[i_labels] for i_data in range(len(data[i_labels]))]

    data, _labels = np.hstack(data), np.hstack(_labels)
    MultiComp = MultiComparison(data, _labels)
    if testfunc is not None:
        # out = MultiComp.allpairtest(scipy.stats.ttest_ind)
        print(MultiComp.allpairtest(testfunc))
    else:
        print(MultiComp.tukeyhsd().summary())

# t_statistic, p_value = scipy.stats.ttest_ind(data1, data2, equal_var=False) # Welch's t test
# W_statistic, p_value = scipy.stats.brunnermunzel(data1, data2)
# H_statistic, p_value = scipy.stats.kruskal(*data) # one-way ANOVA on RANKs


import numpy as np
import scipy.stats as stats

def brunner_munzel_test(x1, x2, distribution='t'):
    '''Calculate Brunner-Munzel-test scores.

       Parameters:
         x1, x2: array_like
           Numeric data values from sample 1, 2.

       Returns:
         w:
           Calculated test statistic.
         p_value:
           Two-tailed p-value of test.
         dof:
           Degree of freedom.
         p:
           "P(x1 < x2) + 0.5 P(x1 = x2)" estimates.

       References:
         * https://oku.edu.mie-u.ac.jp/~okumura/stat/brunner-munzel.html

       Example:
         When sample number N is small, distribution='t' is recommended.

         d1 = np.array([1,2,1,1,1,1,1,1,1,1,2,4,1,1])
         d2 = np.array([3,3,4,3,1,2,3,1,1,5,4])
         print(bmtest(d1, d2, distribution='t'))
         print(bmtest(d1, d2, distribution='normal'))

         When sample number N is large, distribution='normal' is recommended; however,
         't' and 'normal' yield almost the same result.

         d1 = np.random.rand(1000)*100
         d2 = np.random.rand(10000)*110
         print(bmtest(d1, d2, distribution='t'))
         print(bmtest(d1, d2, distribution='normal'))
    '''

    n1, n2 = len(x1), len(x2)
    R = stats.rankdata(list(x1) + list(x2))
    R1, R2 = R[:n1], R[n1:]
    r1_mean, r2_mean = np.mean(R1), np.mean(R2)
    Ri1, Ri2 = stats.rankdata(x1), stats.rankdata(x2)
    var1 = np.var([r - ri for r, ri in zip(R1, Ri1)], ddof=1)
    var2 = np.var([r - ri for r, ri in zip(R2, Ri2)], ddof=1)
    w = ((n1 * n2) * (r2_mean - r1_mean)) / ((n1 + n2) * np.sqrt(n1 * var1 + n2 * var2))
    if distribution == 't':
        dof = (n1 * var1 + n2 * var2) ** 2 / ((n1 * var1) ** 2 / (n1 - 1) + (n2 * var2) ** 2 / (n2 - 1))
        c = stats.t.cdf(abs(w), dof) if not np.isinf(w) else 0.0
    if distribution == 'normal':
        dof = np.nan
        c = stats.norm.cdf(abs(w)) if not np.isinf(w) else 0.0
    p_value = min(c, 1.0 - c) * 2.0
    p = (r2_mean - r1_mean) / (n1 + n2) + 0.5
    return (w, p_value, dof, p)


def calc_partial_corrcoef(x, y, z):
    '''remove the influence of the variable z from the correlation between x and y.'''
    r_xy = np.corrcoef(x, y)
    r_xz = np.corrcoef(x, z)
    r_yz = np.corrcoef(y, z)
    r_xy_z = (r_xy - r_xz*r_yz) / (1-r_xz**2)*(1-r_yz**2)
    return r_xy_z

def nocorrelation_test(x, y, z=None, alpha=0.05):
    if z is None:
        r = np.corrcoef(x, y)[1,0]
    if z is not None:
        r = calc_partial_corrcoef(x, y, z)[1,0]

    n = len(x)
    df = n - 2
    # t = np.abs(np.array(r)) * np.sqrt((df) / (1 - np.array(r)**2))
    t = np.abs(r) * np.sqrt((df) / (1 - r**2))
    t_alpha = stats.t.ppf(1-alpha/2, df)
    p_value = 2*(1-stats.t.cdf(t, df))
    return r, t, p_value
