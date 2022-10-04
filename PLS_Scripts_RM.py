# Rac Mukkamala, White Lab
# Assorted list of PLSR functions I use to automate analysis
# This all came from a Jupyter notebook initially

# Note!!! For all scripts below, both the X and Y matrice must be in the format (n_samples, n_features)
# This is the same standard that is used by scikit-learn.
# Also, most of these functions rely on a pandas dataframe as the input.


### IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from Enrichment_Scripts_RM import run_Enrichr, run_KEA3, run_STRING
from PLSDA_RM import PLSClassifier

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline


def vip_ik_mod(model, feature_labels, X, regex_1, regex_2):
    """
    Regex_1 and Regex_2 is specific to Tig's project and is used to add in the Fold changes for two populations

    model <- PLSR model
    feature_labels <- columns of 
    X <- X phospho matrix
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape

    vips = np.zeros((p,))
    folds = np.zeros((p,))

    X_grp1 = X.filter(regex=regex_1, axis=0)
    X_grp2 = X.filter(regex=regex_2, axis=0)

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        folds[i] = np.median(X_grp1.iloc[:, i]) / np.median(X_grp2.iloc[:, i])

    coef_col = 0
    vips = pd.DataFrame({'VIP': vips, 'Coef':model.coef_[:,coef_col], 
                        'FoldChange': folds, 'AbsLog2FoldChange': np.abs(np.log2(folds)) }, index = feature_labels)
                        
    return vips


def loocv_score_singleY(model, scorer, X, Y):
    """
    Q^2 score for univariate Y matrix.
    model <- sklearn model
    scorer <- scoring function
    X <- X matrix, pandas format only
    Y <- Y matrix, should be a 1D vector/pandas Series.
    """
    loo = LeaveOneOut()
    Y_hat_test = np.zeros(Y.shape)
    train_scores = []

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx, :]
        X_test =  X[test_idx, :]
        Y_train = Y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, Y_train)
        Y_hat_train = model.predict(X_train)
        train_scores.append(scorer(Y_train, Y_hat_train))
        Y_hat_test[test_idx] = model.predict(X_test)
    
    return np.mean(train_scores), scorer(Y, Y_hat_test)



def loocv_score_multiY(model, scorer, X, Y):
    """
    Q^2 score for multivariate Y matrix.
    model <- sklearn model
    scorer <- scoring function
    X <- X matrix, pandas format only
    Y <- Y matrix, should be a 2D matrix/pandas DataFrame, not a vector. For 1D use loocv_score_singleY.
    """
    loo = LeaveOneOut()
    Y_hat_test = np.zeros(Y.shape)
    train_scores = []

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx, :]
        X_test =  X[test_idx, :]
        Y_train = Y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, Y_train)
        Y_hat_train = model.predict(X_train)
        train_scores.append(scorer(Y_train, Y_hat_train))
        Y_hat_test[test_idx,:] = model.predict(X_test)
    
    return np.mean(train_scores), scorer(Y, Y_hat_test)


def PLS_CV(X, Y, model_class=PLSRegression, gs_scoring='neg_mean_squared_error', score_fx=r2_score, cv_range=np.arange(2,20,2), multi_Y=False, verbose=True):
    """
    X <- phospho matrix
    Y <- pheno matrix
    model_class <- type of model to use (PLSRegression or PLSClassifier)
    gs_scoring <- sklearn scoring funcion string name, used in GridSearchCV to find optimal n_components for model
    score_fx <- function to score the performance of model.
    cv_range <- range of values to test for n_components in PLSR
    multi_Y <- True if Y matrix is multivariate
    verbose <- prints out results as function works if True
    """
    if multi_Y: 
        Y = StandardScaler().fit_transform(Y)
    else:
        Y = stats.zscore(Y)
    
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('predictor', model_class())])
    gs = GridSearchCV(estimator=pipe, param_grid={'predictor__n_components':cv_range}, 
                      cv=LeaveOneOut(), scoring=gs_scoring)
    gs.fit(X, Y)

    ncomp = gs.best_params_['predictor__n_components']
    model = model_class(n_components=ncomp)
    if verbose:
        print(f'Best model was {model} with {gs_scoring} {gs.best_score_}')

    if multi_Y:
        train_score, test_score = loocv_score_multiY(model, score_fx, X.to_numpy(), Y)
    else:
        train_score, test_score = loocv_score_singleY(model, score_fx, X.to_numpy(), Y)
    
    if verbose:
        print(f'Train Performance ({score_fx.__name__}): {train_score}')
        print(f'Test Performance ({score_fx.__name__}): {test_score}')

    return model, test_score


# Helper Functions to do enrichment and print out results:

def gene_list_output(vip_table, filename, dir):
    """
    Takes in the VIP table generated by the vips_tig_mod() function and saves 
    them to the given filepath and directory.

    vip_table <- output from vip_ik_mod()
    filename <- name of file where data will be outputted
    dir <- which folder to output the file(s) to.
    """
    os.makedirs(dir, exist_ok=True)
    ct = 0
    with open(f'{dir}/{filename}_genelist.csv', 'w', encoding='utf-8') as out:
        out.write('GeneID,Phosphosite,Sequence,VIP,Coef,FoldChange,Cosine\n')
        for label in vip_table[(vip_table.VIP > 1)].index.to_list():
            out.write(label.replace('_', ',') + ',')
            out.write(str(vip_table.loc[label, 'VIP']) + ',')
            out.write(str(vip_table.loc[label, 'Coef']) + ',')
            out.write(str(vip_table.loc[label, 'FoldChange']) + ',')
            out.write(str(vip_table.loc[label, 'Cosine']) + '\n')
            ct+=1
        print(f'Outputted {ct} genes into {dir}/{filename}_genelist.csv')



default_databases = ['WikiPathway_2021_Human', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020', 'Reactome_2016', 'GO_Biological_Process_2021', 'GO_Molecular_Function_2021',
                'GO_Cellular_Component_2021', 'WikiPathways_2019_Mouse', 'KEGG_2019_Mouse']

def do_enrichment(prot_list, dir, name, databases=default_databases):
    """
    Runs pathway enrichment using Enrichr, KEA, and STRING on the list of proteins inputted.
    Only runs Enrichr and KEA if n > 20 proteins, only runs STRING if n > 5 prots.
    Saves results to given directory.

    prot_list <- list of proteins to input to APIs as a search query
    dir <- which folder to output the file(s) to.
    name <- name of this job, will also be the filename of the output file(s).
    databases <- list of string names of databases in Enrichr to query.
    """
    os.makedirs(dir, exist_ok=True)
    
    print(f'\n\nResults for {name}')
    if len(prot_list) >= 20:
        enrichr_results = run_Enrichr(prot_list, databases, desc=name)
        print('\nTop 10 ENRICHR Results')
        print(enrichr_results.iloc[0:10,[0,1,6]])
        enrichr_results.to_csv(f'{dir}/{name}_enrichr.csv')

        kea_toprank, kea_meanrank = run_KEA3(prot_list, desc=name)
        print('\nTop 5 KEA Results (toprank first then meanrank)')
        print(kea_toprank.iloc[0:5,1:4])
        print(kea_meanrank.iloc[0:5,1:4])
        kea_toprank.to_csv(f'{dir}/{name}_kea.csv')
        kea_meanrank.to_csv(f'{dir}/{name}_kea.csv')
    else:
        print('Less than 20 proteins in this set, omitting Enrichr and KEA')

    if len(prot_list) >= 5:
        print('STRING network')
        run_STRING(prot_list, species_id=10090, img_filepath=f'{dir}/{name}_STRING')
        #this is for displaying the picture in a Jupyter cell.
        #display(Image(filename=f'{dir}/{name}_STRING.png'))

    else:
        print('Less than 5 proteins in this set, omitting STRING')


#this is for a single Y variable
def feature_selection(X, Y, Y_name, N=100):
    '''
    Selects the N (default 100) features with the highest magnitude correlation to the given Y variable.
    X -> X phospho matrix
    Y -> Y phenotypic matrix
    Y_name -> name of column with y variable of interest
    '''
    corrs = []
    for xidx in range(X.shape[1]):
        corrs.append(np.corrcoef(X.iloc[:, xidx], Y.loc[:,Y_name])[0,1])
    
    corrs = np.array(corrs)
    abs_corrs = np.abs(corrs)
    corr_table = pd.DataFrame({'Feature':X.columns, 'Corr':corrs, 'Abs_Corr':abs_corrs})
    corr_table.sort_values(by='Abs_Corr', ascending=False, inplace=True)

    feat_sel = corr_table.iloc[0:N, 0]
    X_sel = X.loc[:, feat_sel]
    return X_sel


def loading_cosines_singleY(model):
    #this is for feature analysis, calculates cosine similarity between feature loadings and pheno loadings.
    #this is particular to single Y variable
    x_load = model.x_loadings_
    y_load = model.y_loadings_

    x_load_norms = np.sqrt(np.sum(x_load**2, axis=1))
    y_load_norm = np.sqrt(np.sum(y_load**2))
    
    cosines = np.sum(x_load * y_load, axis=1)/(x_load_norms*y_load_norm)
    return cosines




def do_plsr_analysis(X, Y, pheno_var, gene_list_folder='./', num_display=20, score_save_cutoff=0.4):
    """
    The mother of all PLSR functions, integrates everything together!
    Takes X, Y, and one phenotypic variable (Y column name) as input.
    Runs PLS_CV on the data to find ideal n_components and print out model test score/Q^2
    Fits full data to the model, and generates some useful plots.
    If CV score > cutoff, model VIPs and data are saved.

    X <- pandas X matrix
    Y <- pandas Y phenotypic matrix
    pheno_var <- name of Y column to isolate for this analysis
    gene_list_folder <- folder where model results will be outputted
    num_display <- number of top VIP score sites to display
    score_save_cutoff <- if test_score > score_save_cutoff, model data/VIPs are saved.
    """

    #feature select only top 100 proteins with highest correlation to Y
    X_sel = feature_selection(X, Y, pheno_var, N=100)
    print(f'Correlation Feature Selection reduced features from {X.shape[1]} to {X_sel.shape[1]}')
    plsr_model, score = PLS_CV(X_sel, Y[pheno_var], cv_range=[2,3,4], multi_Y=False)

    #fit full data to this model, and find VIPs
    phospho_il1b_sel_scaled = StandardScaler().fit_transform(X_sel)
    Y_tot_scaled = stats.zscore(Y[pheno_var])
    plsr_model.fit(phospho_il1b_sel_scaled, Y_tot_scaled)
    vips = vip_ik_mod(plsr_model, X_sel.columns, X_sel, 'EtOH', 'Control')
    vips['Cosine'] = loading_cosines_singleY(plsr_model)

    print(f"{num_display} proteins with VIP > 1 and highest magnitude fold changes listed below")
    vips = vips.sort_values(by='AbsLog2FoldChange', ascending=False)
    print(vips[vips['VIP'] > 1].iloc[0:num_display, [0,1,2,4]])

    #loading plot
    plt.scatter(plsr_model.x_loadings_[vips['VIP']>1,0], plsr_model.x_loadings_[vips['VIP']>1,1], label='X loadings')
    plt.scatter(plsr_model.y_loadings_[:, 0], plsr_model.y_loadings_[:, 1], label='Y loading')
    plt.title('PLSR Loading Plot')
    plt.legend()
    plt.show()

    #feature plot
    sns.scatterplot(x=np.log2(vips[vips.VIP > 1].FoldChange), y=vips[vips.VIP > 1].Cosine, hue=vips[vips.VIP > 1].VIP)
    plt.axvline(x=0, ymin=-1, ymax=1, c='k')
    plt.axhline(y=0, xmin=-3, xmax=3, c='k')
    plt.xlabel('Log2FoldChange EtOH vs Ctrl')
    plt.ylabel('Cosine Similarity to Pheno Var(s)')
    plt.title('Phosphopeptide Feature Plot')
    plt.show()

    #save gene list if score >= cutoff
    if score >= score_save_cutoff:
        print(f'Since CV score = {score} >= {score_save_cutoff}, the FC up and down gene lists will be saved.')
        gene_list_output(vips, pheno_var, f'{gene_list_folder}/{pheno_var}')
        vips['GeneID'] = [name.split('_')[0] for name in vips.index]

        #enrichment combos - this stuff is specific to my project with Tig but gives an idea of how to input specific lists to do_enrichment()

        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcup_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcup_cosdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcdown_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcdown_cosdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_fcdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_cosdown')
        # do_enrichment(vips[(vips.VIP > 1)].GeneID.to_list(), f'{gene_list_folder}/{pheno_var}', f'{pheno_var}_all')

    
    return plsr_model, vips


def feature_selection_multiY(X, Y, Y_names, N=100):
    '''
    Takes each X feature, correlates it to all the Y variables given, and takes the maximum magnitde correlation out of all correlations
    between the given x feature and all Y variables as the scoring metric.

    Chooses the N (default 100) top scoring features for feature selection.
    '''
    corrs = []
    for xidx in range(X.shape[1]):
        corrs_thisX = []
        for Y_name in Y_names:
            corrs_thisX.append(np.corrcoef(X.iloc[:, xidx], Y.loc[:,Y_name])[0,1])
        corrs_thisX = np.abs(corrs_thisX)
        corrs.append(np.max(corrs_thisX))
    
    corrs = np.array(corrs)
    corr_table = pd.DataFrame({'Feature':X.columns, 'Corr':corrs, 'Max_Abs_Corr':corrs})
    corr_table.sort_values(by='Max_Abs_Corr', ascending=False, inplace=True)

    feat_sel = corr_table.iloc[0:N, 0]
    X_sel = X.loc[:, feat_sel]
    return X_sel



def create_combos(list, R):
    """
    Combinations function - creates all combinations of R elements from the inputted list
    """
    if R == 1:
        return [[i] for i in list]
    combos = []
    for i in range(len(list)-R+1):
        elt = list[i]
        rec_result = create_combos(list[i+1:], R-1)
        combos.extend([[elt] + combo for combo in rec_result])
    return combos



def loading_cosines_multiY(model, num_Y):
    #this is for feature analysis, calculates cosine similarity between feature loadings and pheno loadings.
    #this function is particular to multiple Y variable models
    all_cosines = np.zeros((model.n_features_in_, num_Y))
    for i in range(num_Y):
        x_load = model.x_loadings_
        y_load = model.y_loadings_[i, :]

        x_load_norms = np.sqrt(np.sum(x_load**2, axis=1))
        y_load_norm = np.sqrt(np.sum(y_load**2))
        
        cosines_thisY = np.sum(x_load * y_load, axis=1)/(x_load_norms*y_load_norm)
        all_cosines[:, i] = cosines_thisY

    return np.mean(all_cosines, axis=1)



#same as above, just for multiple Y variables.
def do_plsr_analysis_multiY(X, Y, phenos, gene_list_folder='./', num_pheno_sel=5, num_display=20, sel_number=100, score_save_cutoff=0.4):
    """
    The mother of all PLSR functions, integrates everything together! This is for models with multiple Y variables.
    Takes X, Y, and one phenotypic variable (Y column name) as input.
    Runs PLS_CV on the data to find ideal n_components and print out model test score/Q^2
    Fits full data to the model, and generates some useful plots.
    If CV score > cutoff, model VIPs and data are saved.

    X <- pandas X matrix
    Y <- pandas Y phenotypic matrix
    phenos <- list of multiple Y columns to isolate for this analysis
    num_pheno_sel <- number of phenotypes to include in each phenotypic combination of phenos list.
    gene_list_folder <- folder where model results will be outputted
    sel_number <- Number of top X variables with highest correlation to Y matrix to feature select. default = top 100.
    num_display <- number of top VIP score sites to display
    score_save_cutoff <- if test_score > score_save_cutoff, model data/VIPs are saved.
    """

    #Find all num_pheno_sel combinations of Y variables, and fit a PLSR model to each combo. Find the 
    #combo with the highest Q^2 score and use that as the final model
    best_score = 0
    best_pheno_combo = []
    best_model = None
   
    for combo in create_combos(phenos, num_pheno_sel):
        #feature selection - scoring metric is mean correlation across all Y, pick top sel_number features
        X_sel = feature_selection_multiY(X, Y, combo, N=sel_number)
        plsr_model, score = PLS_CV(X_sel, Y[combo], cv_range=[2,3,4], multi_Y=True,verbose=False)

        if score > best_score:
            best_score = score
            best_pheno_combo = combo
            best_model = plsr_model
    
    print(f'Best combo of {num_pheno_sel} Y variables was {best_pheno_combo} with CV Score={best_score}')
    
    #fit best performing model to full data and find VIPs
    X_sel = feature_selection_multiY(X, Y, best_pheno_combo, N=sel_number)
    print(f'Correlation Feature Selection reduced features from {X.shape[1]} to {X_sel.shape[1]}')
    Y_sel = Y[best_pheno_combo]
    
    X_sel_scaled = StandardScaler().fit_transform(X_sel)
    Y_sel_scaled = StandardScaler().fit_transform(Y_sel)
    
    best_model.fit(X_sel_scaled, Y_sel_scaled)
    vips = vip_ik_mod(plsr_model, X_sel.columns, X_sel, 'EtOH', 'Control')
    vips['Cosine'] = loading_cosines_multiY(best_model, num_Y=num_pheno_sel)

    # display top features
    print(f"{num_display} proteins with VIP > 1 and highest magnitude fold changes listed below")
    vips = vips.sort_values(by='AbsLog2FoldChange', ascending=False)
    print(vips[vips['VIP'] >= 1].iloc[0:num_display, :])

    #loading plot
    plt.scatter(best_model.x_loadings_[vips['VIP']>1,0], best_model.x_loadings_[vips['VIP']>1,1], label='X loadings')
    plt.scatter(best_model.y_loadings_[:, 0], best_model.y_loadings_[:, 1], label='Y loading')
    plt.title('PLSR Loading Plot')
    plt.legend()
    plt.show()

    #feature plot
    sns.scatterplot(x=np.log2(vips[vips.VIP > 1].FoldChange), y=vips[vips.VIP > 1].Cosine, hue=vips[vips.VIP > 1].VIP)
    plt.axvline(x=0, ymin=-1, ymax=1, c='k')
    plt.axhline(y=0, xmin=-3, xmax=3, c='k')
    plt.xlabel('Log2FoldChange EtOH vs Ctrl')
    plt.ylabel('Cosine Similarity to Pheno Var(s)')
    plt.title('Phosphopeptide Feature Plot')
    plt.show()

    #save gene list if Q^2 >= cutoff
    if best_score >= score_save_cutoff:
        print(f'Since best CV score = {best_score} >= {score_save_cutoff}, the FC up and down gene lists will be saved.')
        gene_list_output(vips, 'multi', f'{gene_list_folder}/multi')
        vips['GeneID'] = [name.split('_')[0] for name in vips.index]
        
        #enrichment combos - once again this stuff is specific to my project with Tig but gives an idea of how to input specific lists to do_enrichment()
        
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcup_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcup_cosdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcdown_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcdown_cosdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange > 1)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.FoldChange < 1)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_fcdown')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.Cosine > 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_cosup')
        # do_enrichment(vips[(vips.VIP > 1) & (vips.Cosine < 0)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_cosdown')
        # do_enrichment(vips[(vips.VIP > 1)].GeneID.to_list(), f'{gene_list_folder}/multi', 'multi_all')

    return best_model, vips
