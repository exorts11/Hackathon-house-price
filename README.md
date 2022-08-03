# Hackathon-house-price

Comence identificando cuales eran variables categoricas y numericas. Luego rellene todos los valores faltantes.

## Variables numericas
 Los valores nulos encontrados en variables numericas: GarageYrBlt y MasVnrArea se debe a que la feature no existe en la casa,
 por lo que los valores nulos seran reemplazados con el año de construccion de la casa y ceros, respectivamente.

 Los valores nulos en LotFrontage seran reemplazados con la raiz cuadrada de LotArea.

## Variables categoricas 
 En la columna Electrical existe un valor nulo que sera reemplazado con la moda
 Todos los demas valores nulos se deben a que la caracterictica a la que hacen referencia no existe en la casa, por lo que se 
 sustituiran con el valor 'noApply'.
 
## Variables categoricas ordinales 
 Luego cambie los valores de las categoricas ordinales por numeros

##  Correlacion 
Una vez que fueron rellenados procedi a analizar la correlacion entre las variables y el precio de venta. Las analice por bloques,
primero las numericas, luego las categoricas ordinales y las categoricas no ordinales.

Seleccione unicamente las variables que tenian una correlacion mayor a 0.4 y luego, de las variables resultantes,
elimine las que tenian una correlacion mayor a 0.8 entre ellas.

##  Datos atipicos 
Unicamente elimine los registros que tenian datos atipicos en la variable de respuesta, SalePrice, utilizando el rango intercuartil. 

##  Seleccion del modelo
Primero opte por una regresion lineal porque creo que este es un problema de regresion y al ser un dataset relativamente pequeño
mi primera opcion fue resolver la regresion con una ecuacion normal. Fui agregando las avriables por bloque; primero las numericas,
luego las categoricas ordinales y por ultimo las categoricas no ordinales. Apoyandome en todo momento de las curvas de aprendizaje 
para detectar el bias y la varianza de mi modelo.

Para las categoricas no ordinales cree dummies. Luego al agregarlas la ecuacion normal dejo de funcionar porque para algunos samples
en las curvas de aprendizaje no se podia realizar la multiplicacion de las matrices por su semejanza asi que cambie a la libreria de
regresion lineal de sklearn.

Realice un split del set de datos con 60% para entrenamiento y 40% para validacion

##  Optimizacion del modelo
Primero agregue las variables numericas; el modelo tenia un alto bias, entonces agregue features de segundo grado. Cree features 
tales que, para feature1 cree (feature1)^2. Probe con features hasta grado 10 y el mejor era el grado 4.

Luego agregue las variables ordinales categoricas y realice el mismo procedimiento que con las numericas, el valor optimo para el 
grado es 2.

Al final agregue las categoricas no ordinales, y algunas ordinales como no ordinales por lo complejo de convertirlas, y cree dummies 
de ellas. Las agregue al modelo y lo entrene.

# Root Mean Squared Logarithmic Error (RMSLE)
Este modelo presenta un RMSLE de 0.139 en el set de validacion.

##  Diccionario de archivos 
train.ipynb                     ----->      notebook donde se entreno la regresion lineal.

exploracion.ipyn                ----->      archivo de exploracion de datas y desicion de cuales variables usar.

houses.py                       ----->      archivo de funciones que se utilizan a lo largo de todo el proceso.

LinearRegression_house.pkl      ----->      archivo donde se guarda el modelo ya entrenado.

predict.ipybn                   ----->      archivo donde se puede cargar un archivo y realizar una prediccion.
el dataset cargado tiene que ser un data set con las mismas columnas que houses_test_raw.csv.

diferentes_regresiones.ipynb    ----->      archivo donde realice varios tipos de regresion sin ninguna conclusion.

maching_learning.py             ----->      archivo de experimentos fallidos.

pred_test.csv                   ----->      archivo de resultados del house_test_row.csv.

train_gd.ipyn                   ----->      exprimento fallido de regresion lineal sin librerias de sklearn


## Bibliografia
Derivation of the Normal Equation for linear regression.
http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression

Python scikit-learn: exportar clasificador entrenado.
https://foroayuda.es/python-scikit-learn-exportar-clasificador-entrenado/

sklearn.linear_model.LinearRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression#sklearn.linear_model.LinearRegression