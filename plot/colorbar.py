import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)   # 设置子图到下边界的距离

cmap = mpl.colormaps['jet']
norm = mpl.colors.Normalize(vmin=0, vmax=100)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal')
plt.axis('off')
plt.show()