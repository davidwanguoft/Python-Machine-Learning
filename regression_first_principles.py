# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:00:13 2018

@author: david
"""

# r_squared = 1 - sum(y-y_hat)^2 / sum(y_hat_mean)^2


from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

# slope of best fit line is [(x)bar*(y)bar-(xy)bar] / [(x)bar^2 - (x^2)bar]
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    
    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)
print(m)

''' coefficient of determination (r-squared) '''

def squared_error(ys_orig,ys_line):    
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr)/(squared_error_y_mean)

r_squared = coefficient_of_determination(ys,regression_line)
print('r-squared value is: ', r_squared)

    
# draw regression line    

# long form:
'''regression_line = []
for x in xs:
   regression_line.append(m*x + b) '''

regression_line = [(m*x)+b for x in xs]
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.show()


# predictor for y

predict_x = 7

predict_y = (m * predict_x) + b
print("predict_y is: ", predict_y)