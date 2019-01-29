#!/usr/bin/env python3

#print("Hello", "World!")





import numpy as np

import matplotlib.pyplot as plt




"""

#Return the maximum number and the index of that number from an array of numbers

data = [123,-23,454,34,766,-2,0,25,-43,65]

max_value = max(data)
max_index = data.index(max_value)

print("The max value is {0} which is found at index {1}".format(max_value,max_index))






#Plot a sin function as well as noise corrupted data from this sin function (as seen in class)


def sin_function(t):
  
  return np.sin(2*np.pi*t)

t1 = np.arange(0,1,0.1)

y = sin_function(t1)
noise = np.random.normal(0,0.1,10)

data = y + noise # data observed

# origional data
x_coords = np.arange(0,1,0.01)
origional_data = sin_function(x_coords)

plt.plot(t1,data,"bo") # plot observed data
plt.plot(x_coords,origional_data) #plot origional data
plt.title("Plot of noise corrupted data") #give title
plt.legend(["obeserved data",'true underlying function'])
plt.show()
























#Determine if an array is monotonic. 
#Example:

#[1,2,3,3,3,3,3,4,5,6,7] is monotonic (increasing)
#[7,6,5,4,3,3,3,3,3,2,1] is monotonic (decreasing)
#[1,5,4,3,3,3,3,3,2,6,7] is not monotonic



def isMonotonic(arr):
    behaviour = None
    for i in range(len(arr)-1):
        b = arr[i] - arr[i+1]
        if b > 0:
            b = 1
        if b < 0:
            b = -1

#         b = cmp(arr[i], arr[i+1])   cmp cannot be used in python 3 anymore 
        
        
        if b != 0:
            if behaviour == None:
                behaviour = b
            else:
                if behaviour != b:
                    return False
    return True


print (isMonotonic([1,5,4,3,3,3,3,3,2,6,7]))
print (isMonotonic([7,6,5,4,3,3,3,3,3,2,1]))
print (isMonotonic([1,1,1,1,1,1,1]))



"""

















#Hard(er) problem for plotting using matplotlib

#Plot the decision boundary for a circle of radius 2



import matplotlib.colors as colors

size_of_map = np.arange(-3,3,0.01)
xx, yy = np.meshgrid(size_of_map,size_of_map) # gives a rectangular grid out of 
                                              # input values

# our data is generated randomly from the uniform random distribution
x0 = np.random.uniform(-3,3,100)
x1 = np.random.uniform(-3,3,100)

# the underlying classification function, note that it has small gaussian noise
def true_function(x0,x1):
  
  return x0**2 + x1**2 + np.random.normal(0,1) <= 4 

y = true_function(x0,x1) #the true predictions for our data


# our classifier, just the circle without the gaussian noise
def classifier(x0,x1):
  
  return x0**2 + x1**2 <= 4
  
z = classifier(xx,yy) # our classifier applied to the entire grid 


cm_bright = colors.ListedColormap(['#0000FF', '#FF0000']) #just colors
plt.contourf(xx,yy,z,alpha=0.5,cmap=cm_bright) #plot the contours using our classifier results
plt.scatter(x0,x1,c=y,s=20,cmap = cm_bright) #scatter plot of true data

# labelling the plot
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Classifier boundries vs true data results")
plt.show()



size_of_map = np.arange(-3,3,0.01)
xx, yy = np.meshgrid(size_of_map,size_of_map)



print(xx)


print(yy)


plt.plot(xx, yy)
plt.show()










