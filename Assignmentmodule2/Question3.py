
##. Draw the lines y=2 x+1, y=2 x+2, y=2 x+3 in the same figure.



import numpy as numpy
import matplotlib.pyplot as pyplot


x = numpy.linspace(-5, 5, 100)

y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

pyplot.figure()

pyplot.plot(x, y1, color="blue", linstyle='-', label="y1")
pyplot.plot(x, y2, color="red", linestyle='--', label="y2")
pyplot.plot(x, y3, color="green", linestyle='---', label="y3")


pyplot.title("Three different Graph line with y1 as (2x + 1), y2 as (2x +2) and y3 as (2x + 3)")

pyplot.xlabel('x-axis')
pyplot.ylabel('y-axis')

pyplot.legend()
pyplot.grid(True)
pyplot.show()
