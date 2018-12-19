# -*- coding: utf-8 -*-
"""
Created on Thu Dec 06 16:27:40 2018

@author: ragnar
"""

# create example image

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection


n=40
x,y=np.meshgrid(np.arange(n)+1,np.arange(n)+1)


f=plt.figure(figsize=(12,12))
plt.plot(x,y,'ko')
ax=plt.gca()
plt.axis('off')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.axis('equal')
plt.plot()
plt.xlim([0,n+1])
plt.ylim([0,n+1])

ax=plt.gca()
pats=[]
ax.add_patch(pat.Rectangle((0,n/2+2.5),n+1,1,facecolor='r'))
ax.add_patch(pat.Rectangle((0,n/2-0.5),n+1,1,facecolor='g'))
ax.add_patch(pat.Rectangle((0,n/2-3.5),n+1,1,facecolor='b'))

f.savefig("init_image.png", dpi=160, facecolor='w',bbox_inches="tight", pad_inches=0)
plt.show()