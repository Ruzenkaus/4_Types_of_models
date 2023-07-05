from tkinter import Tk, Button, Label, filedialog, Checkbutton, IntVar, messagebox, Entry
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def main_way_to_check():
    selected_path = filedialog.askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))

    if selected_path:
        from_w = loop.e1.get()

        data = pd.read_csv(selected_path)

        if loop.v1.get() != 0 and loop.v2.get() == 0 and loop.v3.get() == 0 and loop.v4.get() == 0:
            X = data.iloc[:, int(from_w):-1].values
            y = data.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
            y_pred = lin_reg.predict(X_test)
            plt.scatter(X, y, color='red')
            plt.plot(X_test, lin_reg.predict(X_test), color='black')
            plt.show()
            loop.l2.config(text=f'{r2_score(y_test, y_pred)}')
        elif loop.v1.get() == 0 and loop.v2.get() != 0 and loop.v3.get() == 0 and loop.v4.get() == 0:

            X = data.iloc[:, int(from_w):-1].values
            y = data.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            regressor = DecisionTreeRegressor(random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            X_grid = np.arange(min(X_test), max(X_test), 0.01)
            X_grid = X_grid.reshape((len(X_grid), 1))
            plt.scatter(X, y, color='red')
            plt.plot(X_grid, regressor.predict(X_grid), color='blue')
            plt.show()
            loop.l2.config(text=f'{r2_score(y_test, y_pred)}')
        elif loop.v1.get() == 0 and loop.v2.get() == 0 and loop.v3.get() != 0 and loop.v4.get() == 0:
            X = data.iloc[:, int(from_w):-1].values
            y = data.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            poly_reg = PolynomialFeatures(degree=4)
            X_poly = poly_reg.fit_transform(X_train)
            regressor = LinearRegression()
            regressor.fit(X_poly, y_train)
            y_pred = regressor.predict(poly_reg.transform(X_test))
            X_grid = np.arange(min(X_test), max(X_test), 0.1)
            X_grid = X_grid.reshape((len(X_grid), 1))
            plt.scatter(X, y, color='red')
            plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')
            plt.show()
            loop.l2.config(text=f'{r2_score(y_test, y_pred)}')


        elif loop.v1.get() == 0 and loop.v2.get() == 0 and loop.v3.get() == 0 and loop.v4.get() != 0:
            X = data.iloc[:, int(from_w):-1].values
            y = data.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            rnd_reg = RandomForestRegressor()
            rnd_reg.fit(X_train, y_train)

            y_pred = rnd_reg.predict(X_test)

            X_grid = np.arange(min(X_test), max(X_test), 0.1)
            X_grid = X_grid.reshape((len(X_grid), 1))
            plt.scatter(X, y, color='red')
            plt.plot(X_grid, rnd_reg.predict(X_grid), color='blue')
            plt.show()

            loop.l2.config(text=f'{r2_score(y_test, y_pred)}')

        else:
            messagebox.showerror('warning', 'Choose only ONE way')


class GUI:
    def __init__(self):
        self.window = Tk()
        self.window.title('Predictor')
        self.window.geometry('800x300')
        self.v1 = IntVar()
        self.v2 = IntVar()
        self.v3 = IntVar()
        self.v4 = IntVar()
        self.e1 = Entry(self.window)

    def main_loop(self):
        l1 = Label(text='Hi! Choose way to predict something with your dataset')
        l1.grid(row=0, column=1)

        c1 = Checkbutton(self.window, text='Multivariable Regression', variable=self.v1, onvalue=1, offvalue=0)
        c1.grid(row=2, column=1)
        c2 = Checkbutton(self.window, text='Decision Tree', variable=self.v2, onvalue=2, offvalue=0)
        c2.grid(row=2, column=2)
        c3 = Checkbutton(self.window, text='Polynomial Regression', variable=self.v3, onvalue=3, offvalue=0)
        c3.grid(row=3, column=1)
        c4 = Checkbutton(self.window, text='Random Forest Regression', variable=self.v4, onvalue=4, offvalue=0)
        c4.grid(row=3, column=2)
        b_s = Button(text='Start prediction', command=main_way_to_check)
        b_s.grid(row=0, column=2)

        self.l2 = Label(text='The accuracy will be there!')
        self.l2.grid(row=5, column=2)
        l3 = Label(text='Print here the start of detecting features')
        l3.grid(row=0, column=4)
        self.e1.grid(row=1, column=4)

        self.window.mainloop()


loop = GUI()

loop.main_loop()
