import numpy as np

def hist(sx):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return list(map(lambda z: float(z)/len(sx), d.values()))

def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*np.log(x)*(-1/np.log(2))

def cmidd(x, y, z):
    return np.sum([a+b-c-d for a,b,c,d in zip(list(entropyd(list(zip(y, z)))),list(entropyd(list(zip(x, z)))),list(entropyd(list(zip(x, y, z)))),list(entropyd(z)))])

# Discrete estimators
def entropyd(sx):
    return  entropyfromprobs(hist(sx))

def entropyfromprobs(probs):
    temp = np.sum(map(elog, probs))
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return temp

def midd(x, y):
    H_X_Y = list(entropyd(list(zip(x, y))))
    H_X = list(entropyd(x))
    H_Y = list(entropyd(y))
    return np.sum([a+b-c for a,b,c in zip(H_X,H_Y,H_X_Y)])

CMIM_INFO = {}
def fast_cmim(X, y, N=5, **kwargs):
    """
    This function implements the CMIM feature selection.
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    t1: {numpy array}, shape: (n_features,)
        minimal corresponding mutual information between selected features and response when 
        conditionned on a previously selected feature

    """
    # Standardize X if CMIM_INFO doesn't exist
    if CMIM_INFO:
        X_mean = CMIM_INFO['X_mean']
        X_std = CMIM_INFO['X_std']
        X = (X - X_mean) / X_std
        X = np.array(X)
        y = np.array(y)
        selected_features = CMIM_INFO['selected_features']
        print(f"Selected features: {selected_features}")
        return X[:, selected_features], y
    else:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        CMIM_INFO['X_mean'] = X_mean
        CMIM_INFO['X_std'] = X_std
    X = (X - X_mean) / X_std
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    is_n_selected_features_specified = False

    n_selected_features = N
    F = np.nan * np.zeros(n_selected_features)
    is_n_selected_features_specified = True

    # t1
    t1 = np.zeros(n_features)
    
    # m is a counting indicator
    m = np.zeros(n_features) - 1
    
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    
    
    for k in range(n_features):
        ### uncomment to keep track
        # counter = int(np.sum(~np.isnan(F)))
        # if counter%5 == 0 or counter <= 1:
        #     print("F contains %s features"%(counter))
        if k == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F[0] = idx
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if np.sum(~np.isnan(F)) == n_selected_features:
                break

        sstar = -1000000 # start with really low value for best partial score sstar 
        for i in range(n_features):
            
            if i not in F:
                
                while (t1[i] > sstar) and (m[i]<k-1) :
                    m[i] = m[i] + 1
                    t1[i] = min(t1[i], cmidd(X[:,i], # feature i
                                             y,  # target
                                             X[:, int(F[int(m[i])])] # conditionned on selected features
                                            )
                               )
                if t1[i] > sstar:
                    sstar = t1[i]
                    F[k+1] = i
                    
    F = np.array(F[F>-100])
    F = F.astype(int)
    t1 = t1[F]
    print(f"Selected features: {F[:n_selected_features]}")
    CMIM_INFO["selected_features"] = F[:n_selected_features]
    return X[:, F[:n_selected_features]], y