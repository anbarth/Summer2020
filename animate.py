import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.linalg import eigh

# script for producing a short mp4 of a SHO+defect potential and its energy levels
# this script was clunky and hard to use
# see makeGif instead, it's better in every way


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-3, 3), ylim=(-18, 15))
Uline, = ax.plot([], [], lw=2)
line0, = ax.plot([], [], lw=2, color="orange")
line1, = ax.plot([], [], lw=2, color="orange")
line2, = ax.plot([], [], lw=2, color="orange")
line3, = ax.plot([], [], lw=2, color="orange")
line4, = ax.plot([], [], lw=2, color="orange")
line5, = ax.plot([], [], lw=2, color="orange")
line6, = ax.plot([], [], lw=2, color="orange")

left = -15
right = 15
dx = 0.05
D = int((right-left)/dx)
domain = np.linspace(left,right,D,endpoint=False)

plotLeft = -4
plotRight = 4
mid = int(D/2)
wing = int(plotLeft/left/2 * D)
domain_r = domain[mid-wing:mid+wing]
D_r = len(domain_r)


# initialization function: plot the background of each frame
def init():
    Uline.set_data([],[])
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    line6.set_data([], [])
    #return Uline,line0,
    return Uline,line0,line1,line2,line3,line4,line5,line6,


# animation function.  This is called sequentially
def animate(i):

    # define U
    U = np.zeros((D))
    x = left
    for j in range(D):
        pot = x*x
        if x <= 1 and x >= -1:
            pot -= 15*np.abs(np.sin(0.01*i))
        U[j] = pot
        x += dx
    Uline.set_data(domain_r, U[mid-wing:mid+wing])

    # construct H
    ham = np.zeros((D,D))
    for i in range(D):
        # diagonal terms: U (potential)+2/dx^2 (kinetic)
        ham[i][i] = U[i]+2/(dx*dx)
        # tridiagonal terms: -1/dx^2 (kinetic)
        if i > 0:
            ham[i][i-1] = -1/(dx*dx)
            ham[i-1][i] = -1/(dx*dx)

    E = eigh(ham,eigvals=(0,7),eigvals_only=True) #TODO only need Es
    line0.set_data(domain_r,E[0])
    line1.set_data(domain_r,E[1])
    line2.set_data(domain_r,E[2])
    line3.set_data(domain_r,E[3])
    line4.set_data(domain_r,E[4])
    line5.set_data(domain_r,E[5])
    line6.set_data(domain_r,E[6])

    return Uline,line0,line1,line2,line3,line4,line5,line6,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=314, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('defect-eigens.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()