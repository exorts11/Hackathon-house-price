import pandas as pd
import re
import numpy as np

def countNull (data):
    # Porcentaje de valores nulos en cada columna
    n = pd.DataFrame(data.isnull().sum(), columns=['Nulos'])
    n['Entries'] = data.shape[0]
    
    n['Porcentaje'] = n['Nulos']/n['Entries']

    return n

def outliers_obt(data, columna,valoriqr=1.5):
    ##calculamos los cuartiles 
    Q1 = data[columna].quantile(0.25)
    #print('Primer Cuartile', Q1)
    Q3 = data[columna].quantile(0.75)
    #print('Tercer Cuartile',Q3)
    IQR = Q3 - Q1
    #print('Rango intercuartile', IQR)

    ##calculamos los bigotes superior e inferior
    BI = (Q1 - valoriqr * IQR)
    #print('bigote Inferior \n', BI)
    BS = (Q3 + valoriqr * IQR)
    #print('bigote superior \n', BS)

    ##obtenemos una nueva tabla sin los outliers
    ubi_sin_out = data.loc[(data[columna] >= BI) & (data[columna] <= BS),:].index
    return ubi_sin_out


def transform(df, pre):
    # Obtener feaures
    columns = []
    #features numericas
    numerical = []
    n = 0
    count = 0

    with open("datasets/readme.txt", "r") as file:
        for line in file:
            if re.findall('[a-zA-Z0-9]+:\s+[a-zA-Z]+', line):
                columns.append(re.findall('([a-zA-Z0-9]+):\s+[a-zA-Z]+', line)[0])
                count+=1
                #print(n)

                if n<3:
                    numerical.append(count-2)
                    #print(count-2, columns[count-2])

                n = 0

            n+=1

    z = np.array(columns)

    numerical = z[numerical]
    numerical = np.append (numerical, ['BedroomAbvGr', 'KitchenAbvGr'])
    indexes = ['Bedroom', 'Kitchen']
    numerical = np.setdiff1d(numerical, indexes)

    col = np.append (z, ['BedroomAbvGr', 'KitchenAbvGr', 'MiscVal'])
    indexes = ['Bedroom', 'Kitchen']
    col = np.setdiff1d(col, indexes)

    nulls = countNull(df)
    
    # variables numericas con nulos
    mask = []
    for feature in numerical:
        if feature in nulls.loc[nulls.Porcentaje > 0, 'Porcentaje']:
            #print(feature)
            mask.append(feature)

    #-------------------TRANSFORMACION DE DATOS-------------------
    # GarageYrBlt y MasVnrArea
    df.loc[:,mask] = df.loc[:,mask].fillna(0)
    df.loc[df.GarageYrBlt == 0, 'GarageYrBlt'] = df.loc[df.GarageYrBlt == 0, 'YearBuilt']

    # LotFrontage
    df.loc[df.LotFrontage == 0, 'LotFrontage'] = df.loc[df.LotFrontage == 0, 'LotArea']**(1/2)

    # Electrical
    df.loc[df.Electrical.isnull(),'Electrical'] = df.loc[:,'Electrical'].mode()

    # Todas las demas columnas
    df.fillna('noApply', inplace=True)

    #---------------------------NUMERICAL VARIABLES-----------------------
    #z = np.append(numerical, 'SalePrice')
#
    #corr = df.loc[:, z].corr()
#
    #mask = z[corr.SalePrice.values < 0.4]
    #corr.drop(columns=mask,axis=1, inplace= True)
    #corr.drop(mask, axis=0, inplace=True)

    #-------------------VARIABLES CATEGORICAS ORDINALES-------------------
    cat_ord = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
    'PoolQC']

    quality = pd.DataFrame(['noApply', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], columns=['name'])
    quality.reset_index(inplace=True)
    quality.rename(columns = {'index':'id'}, inplace = True)

    #new_df = df.copy()

    cat = []
    for feature in cat_ord:
        if df[feature].unique()[0] in quality.name.unique():
            df[feature+'key'] = pd.merge(right = quality, left= df[feature], how='left', left_on=feature, right_on='name').id
            df.drop(columns=[feature], inplace=True)
            cat.append(feature)

    #-----------------------------VARIABLES A USAR---------------------------
    cat_ord = ['ExterQualkey', 'BsmtQualkey', 'HeatingQCkey', 'KitchenQualkey',
       'FireplaceQukey']

    numerical1 = ['Fireplaces', 'GarageCars',
       'GrLivArea', 'MasVnrArea', 'TotalBsmtSF',
       'YearBuilt', 'YearRemodAdd']

    categorical_no_ord = ['BsmtFinType1', 'Foundation', 'GarageFinish',
        'Neighborhood', 'OverallQual']


    #--------------------VARIABLES CATEGORICAS NO ORDINALES
    
    #li_not_plot = list(numerical) + cat
    #li_feats = [c for c in list(col) if c not in li_not_plot]

    #----------Preparando dataframe
    df = df.loc[:, numerical1 + cat_ord + categorical_no_ord + ['SalePrice']]

    #------------------------------------
    df.loc[:, 'OverallQual'] = (df.loc[:, 'OverallQual'] + 100).apply(chr)
    #df.loc[:, 'OverallCond'] = (df.loc[:, 'OverallCond'] + 100).apply(chr)
    #df.loc[:, 'MSSubClass'] = ((df.loc[:, 'MSSubClass']/10).astype('int') + 100).apply(chr)

    # Get dummies
    for feature in categorical_no_ord:
        #convertimos variables en string y las identificamos con el nombre de la feature
        df.loc[:, feature] = df.loc[:,feature].astype('str')
        df.loc[:, feature] = feature + '_' +  df.loc[:, feature]
        # Creamos los dummies
        df = pd.concat([df, pd.get_dummies(df[feature])], axis=1)
        df.drop(columns=feature, axis=1, inplace=True)

    # Hacemos una transformacion logaritmica a la variable de respuesta y luego eliminamos outlayers
    if pre == 0:
        df1 = df.copy()
        df1.loc[:, 'SalePrice'] = np.log(df.SalePrice).copy()
        df = df.loc[outliers_obt(df1, 'SalePrice'),:].reset_index().copy()
        df.drop(columns='index', inplace=True)

    return df, numerical1, cat_ord#corr.index[corr.index != 'SalePrice']


def normalEqn(X, y):
    theta = np.zeros((X.shape[0],1))

    z = np.linalg.inv(np.matmul(X.transpose(),X))

    z1 = np.matmul(z,X.transpose())

    theta = np.matmul(z1,y)

    return theta

#function [theta] = normalEqn(X, y)
#%NORMALEQN Computes the closed-form solution to linear regression 
#%   NORMALEQN(X,y) computes the closed-form solution to linear 
#%   regression using the normal equations.
#
#theta = zeros(size(X, 2), 1);
#
#% ====================== YOUR CODE HERE ======================
#% Instructions: Complete the code to compute the closed form solution
#%               to linear regression and put the result in theta.
#%
#
#% ---------------------- Sample Solution ----------------------
#
#Z =  inv(X'*X);
#theta = Z*X'*y;
#
#% -------------------------------------------------------------
#
#
#% ============================================================
#
#end

def rmsle (predict, actual):

    m = predict.shape[0]
    z = (np.log(predict+1) - np.log(actual+1))**2
    
    rm = (z.sum()/m)**(1/2)

    return rm

def testError(predict, actual):
    z = (predict - actual)**2
    z = z.sum()/(2*predict.shape[0])

    return z


def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for idx, i in enumerate(range(1, p+1)):
        X_poly[:,idx] = X**i
    return pd.DataFrame(X_poly)

#function [X_poly] = polyFeatures(X, p)
#%POLYFEATURES Maps X (1D vector) into the p-th power
#%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
#%   maps each example into its polynomial features where
#%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
#%
#
#
#% You need to return the following variables correctly.
#X_poly = zeros(numel(X), p);
#
#% ====================== YOUR CODE HERE ======================
#% Instructions: Given a vector X, return a matrix X_poly where the p-th 
#%               column of X contains the values of X to the p-th power.
#%
#% 
#
#for i=1 : p
#    
#    X_poly(:,i) = X.^i;
#    
#end
#
#
#
#
#% =========================================================================
#
#end

def featureNormalize(X):
    m = X.shape[1]
    X_nom = X.copy()
    mu = np.zeros((m,1))
    sigma = np.zeros((m,1))
    for idx, feature in enumerate(X.columns):
        z = X.loc[:,feature]
        y = z.mean()
        w = z.std()
        
        X_nom.loc[:,feature] = (X_nom.loc[:,feature].copy() - y)/w

        mu[idx] = y
        sigma[idx] = w

    return X_nom, mu, sigma

#function [X_norm, mu, sigma] = featureNormalize(X)
#%FEATURENORMALIZE Normalizes the features in X 
#%   FEATURENORMALIZE(X) returns a normalized version of X where
#%   the mean value of each feature is 0 and the standard deviation
#%   is 1. This is often a good preprocessing step to do when
#%   working with learning algorithms.
#
#% You need to set these values correctly
#X_norm = X;
#mu = zeros(1, size(X, 2));
#sigma = zeros(1, size(X, 2));
#
#% ====================== YOUR CODE HERE ======================
#% Instructions: First, for each feature dimension, compute the mean
#%               of the feature and subtract it from the dataset,
#%               storing the mean value in mu. Next, compute the 
#%               standard deviation of each feature and divide
#%               each feature by it's standard deviation, storing
#%               the standard deviation in sigma. 
#%
#%               Note that X is a matrix where each column is a 
#%               feature and each row is an example. You need 
#%               to perform the normalization separately for 
#%               each feature. 
#%
#% Hint: You might find the 'mean' and 'std' functions useful.
#%       
#
#for i = 1 : size (X, 2)
#    
#    z = X(:,i);
#    y = mean(z);
#    w = std(z);
#    for h = 1 : length(X)
#       
#        X_norm(h,i) = (X_norm(h,i)-y)/w;
#        
#    end
#    
#    mu(1,i) = y;
#    sigma(1,i) = w;
#    
#end



#function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
#%GRADIENTDESCENT Performs gradient descent to learn theta
#%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#%   taking num_iters gradient steps with learning rate alpha
#
#% Initialize some useful values
#% number of training examples
#m = length(y);
#J_history = zeros(num_iters, 1);
#
#for iter = 1:num_iters
#
#    % ====================== YOUR CODE HERE ======================
#    % Instructions: Perform a single gradient step on the parameter vector
#    %               theta. 
#    %
#    % Hint: While debugging, it can be useful to print out the values
#    %       of the cost function (computeCost) and gradient here.
#    %
#    
#    predicciones = X*theta;
#    Errores = (predicciones-y);
#    x1 = X(:, 2);
#    t0 = theta(1,1) - (alpha/m)*sum(Errores);
#    t1 = theta(2,1) - (alpha/m)*sum(Errores.*x1);
#    theta(1,1) = t0;
#    theta(2,1) = t1;
#    
#    
#    % ============================================================
#
#    % Save the cost J in every iteration    
#    J_history(iter) = computeCost(X, y, theta);
#
#end
#
#end
#
#
#
#function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
#%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#%   taking num_iters gradient steps with learning rate alpha
#
#% Initialize some useful values
#m = length(y); % number of training examples
#J_history = zeros(num_iters, 1);
#tn = theta;
#xn = zeros(length(X),1);
#
#for iter = 1:num_iters
#
#    % ====================== YOUR CODE HERE ======================
#    % Instructions: Perform a single gradient step on the parameter vector
#    %               theta. 
#    %
#    % Hint: While debugging, it can be useful to print out the values
#    %       of the cost function (computeCostMulti) and gradient here.
#    %
#    predicciones = X*theta;
#    Errores = (predicciones-y);
#    
#    for i=1:length(theta)
#        xn = X(:,i);
#        tn(i,1) = theta(i,1) - (alpha/m)*sum(Errores.*xn);
#    end
#    
#    theta = tn;
#    
#    % ============================================================
#
#    % Save the cost J in every iteration    
#    J_history(iter) = computeCostMulti(X, y, theta);
#
#end
#
#end