# This code is an introduction to basic graphing using matplot

import matplotlib.pyplot as plt

x = [1,2,3] # x axis values 
y = [1.5,1.46,1.77] # corresponding y axis values

plt.plot(x, y) # plotting the points 
plt.xlabel('x - axis') # naming the x axis
plt.ylabel('y - axis') # naming the y axis 
plt.title('My first graph!') # giving a title to my graph 
plt.show() # function to show the plot 
