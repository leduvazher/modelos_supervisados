
######################################################################
#                          Pre-procesamiento                         #
######################################################################
#
# Paquetes requeridos: AppliedPredictiveModeling, e1071, caret y 
# corrplot
# 
# Limpiar el ambiente
#

rm(list=ls())

# Establecer working directory
#
#setwd("C:\\Users\\miguel.villalobos\\Dropbox\\AplicacionModPredConR\\Datos")
# 
# Cargar paquetes requeridos
#
if(!require(dplyr)) install.packages("dplyr")
library(dplyr)
if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)
if(!require(AppliedPredictiveModeling)) 
  install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(e1071)) install.packages("e1071")
library(e1071)
if(!require(corrplot)) install.packages("corrplot")
library(corrplot)
library(AppliedPredictiveModeling)
data(segmentationOriginal)
names(segmentationOriginal)
head(segmentationOriginal$Case)
str(segmentationOriginal)
dim(segmentationOriginal)

# Separamos el archivo de entrenamiento

segTrain = subset(segmentationOriginal, Case == "Train")
dim(segTrain)
# Eliminamos las 3 primeras columnas (identificacdores)
# y los guardamos en vectores separados
head(segTrain$Class)
segTrainClass <- segTrain$Class
segTrainX <- segTrain[, -(1:3)]
dim(segTrainX)

#
# Los datos oroginales contienen varias columnas de "Status" que
# son versiones binarias de los predictores.  Para removerlas
# encontramos los nombres de columnas que contienen "Status" y
# las removemos.  Esto lo usamos usando el comando "grep" que busca
# un patrón en cada elemento de un vector de caracteres.

statusColNum = grep("Status",names(segTrainX))
head(statusColNum,10)
dim(segTrainX)
segTrainX = segTrainX[, -statusColNum]
dim(segTrainX)

######################################################################
#              Transformación de predictores                         #
###################################################################### 

# Las variable VarIntenChX miden la desviación estándar de la 
# intensidad de los pixeles en los filamentos activos correspondientes
# 
# Por ejemplo veamos como se distribuye VarIntenCh3

###Para estos modelos el sesgo dificulta el entramiento del modelo

max(segTrainX$VarIntenCh3)/min(segTrainX$VarIntenCh3)

library(e1071)
skewness(segTrainX$VarIntenCh3)
histogram(segTrainX$VarIntenCh3, col="brown")

# Como todos los predictores son numéricos, podemos utilizar la 
# función apply para calcular skewwnes en las columnas
# apply(X,MARGIN,FUN), donde MARGIN = 2 significa que la función
# FUN se aplica a las columnas

skewValues = apply(segTrainX, 2, skewness)
head(skewValues)
names(skewValues)

# Utilizando los valores de skewness podemos priorizar las variables
# para visualizar su distribución.
#
# El paquete MASS contiene la función boxcox, la cual se puede utilizar
# para estimar lambda, sin embargo no crea las variables transformadas.
# En cambio una función en el paquete caret, "BoxCoxTrans" puede
# encontrar la transformación appropiada y aplicarla a los datos.

library(caret)
VarIntenCh3trans = BoxCoxTrans(segTrainX$VarIntenCh3)
VarIntenCh3trans

# Después de la transformación
Varint3.tr = predict(VarIntenCh3trans, 
                     segTrainX$VarIntenCh3)

head(Varint3.tr, 10)
head(segTrainX$VarIntenCh3)
head(cbind(segTrainX$VarIntenCh3,Varint3.tr ), 10)
skewness(Varint3.tr)
histogram(Varint3.tr, col="brown")

# Usamos la función preProcess de caret para transformar la variable
segPP <- preProcess(segTrainX, method = "BoxCox")
names(segPP)
names(segPP$bc)
#str(segPP$bc)

# Aplicamos la transformación
segTrainTrans <- predict(segPP, segTrainX)

# Resultados para VarIntCh3
segPP$bc$VarIntenCh3

histogram(segTrainX$VarIntenCh3,
          xlab = "Natural Units",
          type = "count", col="brown")

histogram(log(segTrainX$VarIntenCh3),
          xlab = "Log Units",
          ylab = " ",
          type = "count", col="brown")

# Resultados para PerimCh1
segPP$bc$PerimCh1$lambda

histogram(~segTrainX$PerimCh1,
          xlab = "Natural Units",
          type = "count", col="brown")

histogram(~segTrainTrans$PerimCh1,
          xlab = "Transformed Data",
          ylab = " ",
          type = "count", col="brown")

#
# Retomar presentación
#

######################################################################
#                     Removiendo variables                           #
###################################################################### 

# Filtrando variables con varianza cercana a cero
# Para filtrar predictores con varianza cercana a acero, utilizamos
# la función nearZeroVar, la cual regresa los números de columnas
# que tienen casi cero varianza

nzvar = nearZeroVar(segTrainTrans)
nzvar


# Filtrando predictores altamente correlacionados

segCorr <- cor(segTrainTrans)
str(segCorr)
dim(segCorr)
segCorr

library(corrplot)
corrplot(segCorr, order = "hclust", tl.cex = .35)

# Usamos la función findCorrelation para identificar las columnas
# a remover
#
highCorr <- findCorrelation(segCorr, .75)
highCorr
str(highCorr)

# Ahora las eliminamos del conjunto de entrenamiento
segTrainXsinCorr = segTrainTrans[-highCorr]
dim(segTrainX)
dim(segTrainXsinCorr)

#
# Regresar a presentación
#

######################################################################
# Creación de variables indicadoras (dummy) para predictores         #
# categóricos.                                                       #
######################################################################
#
# Creación de variables indicadoras: En ocasiones crear variables 
# dummy es útil.  Por ejemplo, los nodos en un modelo de árbol son 
# más fácilmente interpretables cuando variables dummy codifican toda
# la información para un predictor.  Es recomendable utilizar 
# conjuntos completos de variables dummy cuando se trabaja con 
# algoritmos de árbol.
#
# Tomemos un subconjunto del conjunto de datos cars que viene con el
# paquete caret.  Para 2005, se tienen datos de re-venta para 804 
# autos GM.  El objetivo del modelo es predecir el precio del auto 
# basado en características conocidas.  Nos enfocaremos en precio, 
# millaje, y tipo de auto.

data(cars)
head(cars)
names(cars)
str(cars)
carSubset = cars[,-c(3:5,7:12)]
names(carSubset)
# 
# O bien utilizando la función select de dplyr
#
carSubset = cars %>% select(Price,Mileage,Sound,Saturn, convertible,
                            coupe, hatchback, sedan, wagon)
names(carSubset)

head(carSubset)

type <- c("convertible", "coupe", "hatchback", 
          "sedan", "wagon")
carSubset$Type <- factor(apply(carSubset[, 5:9], 
                               1,function(x) 
                                 type[which(x == 1)]))

head(cbind(carSubset[,5:9], carSubset$Type))
str(carSubset$Type)

# Para modelar el precio como una función del millaje y tipo de auto, 
# podemos utilizar la función dummyVars para determinar la 
# codificación para los predictores.
#
# Supongamos que nuestro primer modelo asume que el precio se puede 
# modelar como un modelo aditivo del millaje y el tipo:

simpleMod = dummyVars(~Mileage + Type, 
                      data = carSubset, levelsOnly=TRUE)


simpleMod

# Para generar las variables dummy para el conjunto de entrenamiento o
# para nuevas muestras, el método de predict se usa en conjunto con el
# objeto de dummyVars

predict(simpleMod, head(carSubset))

# Si pensamos que pudiera haber una interacción entre el tipo y el millaje, agregamos
# el término Mileage:Type

withInteraction = 
  dummyVars(~Mileage + Type + Mileage:Type, 
            data=carSubset,levelsOnly=TRUE)
withInteraction
predict(withInteraction, head(carSubset))

######################################################################