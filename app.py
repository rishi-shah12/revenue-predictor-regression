import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def apple_prediction():
    dfApple = pd.read_csv('apple.csv')

    appleX = dfApple.iloc[:, 0:1].values
    appley = dfApple.iloc[:, 1].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(appleX)

    poly.fit(X_poly, appley)
    lin2 = LinearRegression()
    lin2.fit(X_poly, appley)

    plt.scatter(appleX, appley, color='blue')


    plt.plot(appleX, lin2.predict(poly.fit_transform(appleX)), color='red')
    plt.title('Polynomial Regression stock price')
    plt.xlabel('Year')
    plt.ylabel('Revenue ($)')

    plt.show()

    print("Apple 2020 Revenue Difference Prediction vs Actual (%):")
    print(abs((lin2.predict(poly.fit_transform([[2020]]))) - 274515) / 274515 * 100)

def amzn_prediction():
    dfAmzn = pd.read_csv('amzn.csv')

    amznX = dfAmzn.iloc[:, 0:1].values
    amzny = dfAmzn.iloc[:, 1].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(amznX)

    poly.fit(X_poly, amzny)
    lin2 = LinearRegression()
    lin2.fit(X_poly, amzny)

    plt.scatter(amznX, amzny, color='blue')

    plt.plot(amznX, lin2.predict(poly.fit_transform(amznX)), color='red')
    plt.title('Polynomial Regression stock price')
    plt.xlabel('Year')
    plt.ylabel('Revenue ($)')

    plt.show()
    print("Amazon 2019 Revenue Difference Prediction vs Actual (%):")
    print(abs((lin2.predict(poly.fit_transform([[2019]]))) - 280522) / 280522 * 100)


def msft_prediction():
    dfMsft = pd.read_csv('msft.csv')

    msftX = dfMsft.iloc[:, 0:1].values
    msfty = dfMsft.iloc[:, 1].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(msftX)

    poly.fit(X_poly, msfty)
    lin2 = LinearRegression()
    lin2.fit(X_poly, msfty)

    plt.scatter(msftX, msfty, color='blue')

    plt.plot(msftX, lin2.predict(poly.fit_transform(msftX)), color='red')
    plt.title('Polynomial Regression stock price')
    plt.xlabel('Year')
    plt.ylabel('Revenue ($)')

    plt.show()
    print("Microsoft 2020 Revenue Difference Prediction vs Actual (%):")
    print(abs((lin2.predict(poly.fit_transform([[2020]]))) - 143015) / 143015 * 100)

apple_prediction()
msft_prediction()
amzn_prediction()