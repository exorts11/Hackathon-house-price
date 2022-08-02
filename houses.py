import pandas as pd
import re
import numpy as np

def countNull (data):
    # Porcentaje de valores nulos en cada columna
    n = pd.DataFrame(data.isnull().sum(), columns=['Nulos'])
    n['Entries'] = data.shape[0]
    
    n['Porcentaje'] = n['Nulos']/n['Entries']

    return n

def transform(df):
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
    df.loc[df.Electrical.isnull(),'Electrical'] = 'SBrkr'

    # Todas las demas columnas
    df.fillna('noApply', inplace=True)

    #---------------------------NUMERICAL VARIABLES-----------------------
    z = np.append(numerical, 'SalePrice')

    corr = df.loc[:, z].corr()

    mask = z[corr.SalePrice.values < 0.4]
    corr.drop(columns=mask,axis=1, inplace= True)
    corr.drop(mask, axis=0, inplace=True)

    return df, corr.index


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