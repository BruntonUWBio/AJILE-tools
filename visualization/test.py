__author__      = "Nancy Xin Ru Wang"

import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

npdata = numpy.random.randint(100, size=(5,6,10))
plotlays, plotcols = [2,5], ["black","red"]

fig = plt.figure()
ax = plt.axes(xlim=(0, numpy.shape(npdata)[0]), ylim=(0, numpy.max(npdata)))
timetext = ax.text(0.5,50,'')

lines = []
for index,lay in enumerate(plotlays):
    lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)

def init():
    for line in lines:
        line.set_data([],[])
    return lines

def animate(i):
    timetext.set_text(i)
    x = numpy.array(range(1,npdata.shape[0]+1))
    for lnum,line in enumerate(lines):
        line.set_data(x,npdata[:,plotlays[lnum]-1,i])
    return lines


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=numpy.shape(npdata)[1], interval=100, blit=True)

plt.show()