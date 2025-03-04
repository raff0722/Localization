import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import imageio
import os
from pathlib import Path, PurePath
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from Localization.plots_conf import SISC
from Localization.twoD_deblurr_image import functions
from Localization.twoD_deblurr_image.Problem_data import computing_times

plot_dir = PurePath(r'C:\Users\raff\Projects\Localization\Overleaf\SISC\main_6\Graphics')
data_dir = PurePath( r'twoD_deblurr_image\Problem_data' )

#%% Figure 3: complete image with sections and data

[x_im, y_im_true, y_im, lam, N, d] = functions.load(data_dir / 'conf8' / 'problem')
par = functions.load(data_dir / 'conf8' / 'par')

q = 64

SISC(grid=0, fig_height=7, fig_width=13)

fig = plt.figure()

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

im = grid[0].imshow(x_im, vmin=0, vmax=par['max_int'], cmap='gray')

grid[0].vlines([q*ii-0.5 for ii in range(1,8)], 0-0.5, 512-0.5, lw=0.8, color='white')
grid[0].hlines([q*ii-0.5 for ii in range(1,8)], 0-0.5, 512-0.5, lw=0.8, color='white', label=r'$64\times 64$')

len_sect = [128, 256, 384, 512]
corner = [3*q-0.5, 2*q-0.5, 1*q-0.5, 0*q-0.5]
lst = ['dashdot', 'dotted', 'dashed', 'solid']
leg = [r'$128\times128$', r'$256\times256$', r'$384\times384$', r'$512\times512$', ]

for ii in range(len(len_sect)):
    # rect = patches.Rectangle((corner[ii], corner[ii]), len_sect[ii], len_sect[ii], linewidth=0.8, edgecolor='red', facecolor='none', ls=lst[ii], zorder=100, label=leg[ii])
    rect = patches.Rectangle((corner[ii], corner[ii]), len_sect[ii], len_sect[ii], linewidth=0.8, edgecolor='black', facecolor='none', ls=lst[ii], zorder=100, label=leg[ii])
    grid[0].add_patch(rect)

grid[0].set_axisbelow(True)
l5 = grid[0].legend(bbox_to_anchor=(0.055, 0), loc="lower left",
                bbox_transform=fig.transFigure, ncol=3)

grid[0].set_title(r'True image with block partition')
grid[0].set_yticks([])
grid[0].set_xticks([])

im = grid[1].imshow(y_im, vmin=0, vmax=par['max_int'], cmap='gray')
grid[1].set_axisbelow(True)
grid[1].set_title(r'Data')
grid[1].set_yticks([])
grid[1].set_xticks([])

# colorbar
grid[1].cax.colorbar(im)
grid[1].cax.toggle_label(True)

plt.tight_layout()
plt.savefig(plot_dir / 'compl_sect_noCol.pdf', dpi=1000)
# plt.savefig(plot_dir / 'compl_sect_ex1.pdf', dpi=1000)

#%% Figure 4: MAP, mean and CI difference for eps=1e-3, 1e-5, 1e-7

eps = ['3', '5', '7']
# sect = np.ix_(np.arange(50,264), np.arange(160,374))

par = functions.load(data_dir / 'conf8' / 'par')
[x_im, y_im_true, y_im, lam, N, d] = functions.load(data_dir / 'conf8' / 'problem')

vmin_CI = 0
vmax_CI = 0.28

fig = plt.figure()
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(3,3),
                 axes_pad=0.50,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

for ii, eps_ii in enumerate(eps):

    map_path = data_dir / 'conf8' / ('sam_fM_eps'+eps_ii) / 'map_eps'
    map = functions.load( map_path)

    stats_path = data_dir / 'conf8' / ('sam_lM_eps'+eps_ii) / 'stats'
    stats = functions.load( stats_path )

    mean = stats['mean']
    CI_l = stats['CI'][:, 0]
    CI_u = stats['CI'][:, 1]

    im = grid.axes_row[ii][0].imshow(functions.res(map, (N,N)), vmin=0, vmax=par['max_int'], cmap='gray')
    grid.axes_row[ii][0].set_axisbelow(True)
    grid.axes_row[ii][0].cax.colorbar(im)
    grid.axes_row[ii][0].cax.toggle_label(True)

    im = grid.axes_row[ii][1].imshow(functions.res(mean, (N,N)), cmap='gray', vmin=0, vmax=par['max_int'])
    grid.axes_row[ii][1].set_axisbelow(True)
    grid.axes_row[ii][1].cax.colorbar(im)
    grid.axes_row[ii][1].cax.toggle_label(True)

    im = grid.axes_row[ii][2].imshow(functions.res(CI_u-CI_l, (N,N)), cmap='gray', vmin=vmin_CI, vmax=vmax_CI)
    grid.axes_row[ii][2].set_axisbelow(True)
    grid.axes_row[ii][2].cax.colorbar(im)
    grid.axes_row[ii][2].cax.toggle_label(True)

    grid.axes_row[ii][0].set_ylabel(r'$\varepsilon=10^{-'+eps_ii+'}$')    
    grid.axes_row[ii][0].set_xticks([])
    grid.axes_row[ii][0].set_yticks([])
    
grid.axes_row[0][0].set_title(r'MAP estimate')
grid.axes_row[0][1].set_title(r'Sample mean')
grid.axes_row[0][2].set_title(r'90% CI difference')

plt.tight_layout()
plt.savefig(plot_dir / 'mean_CI_eps.pdf', dpi=1000)

#%% Figure 5: local acceptance rates of blocks for fixed step size

# y axis starts at top
# x axis starts at left

SISC(grid=0, fig_height=9, fig_width=10)

fig, ax = plt.subplots()

q = 64
n_ch = 5

ax.imshow(np.zeros((512, 512)), vmin=0, vmax=1, cmap='binary') #, extent=())

ax.vlines([q*ii-0.5 for ii in range(0,9)], 0-0.5, 512-0.5, lw=1, color='k')
ax.hlines([q*ii-0.5 for ii in range(0,9)], 0-0.5, 512-0.5, lw=1, color='k')

corner = [3*q-0.5, 2*q-0.5, 1*q-0.5, 0*q-0.5] # lower left corners of image sections
len_sect = [128, 256, 384, 512] # lengths of image sections

# lst = ['dashdot', 'dotted', 'dashed', 'solid']
lst = ['solid', 'solid', 'solid', 'solid']

# for writing acceptance rates in blocks
y_tex_corr = 17
y_tex_pad = 2
x_tex_corr = -9

for ii in range(4):
    
    # red rectangles (image sections)
    # rect = patches.Rectangle((corner[ii], corner[ii]), len_sect[ii], len_sect[ii], linewidth=1, edgecolor='red', facecolor='none', ls=lst[ii], zorder=100) #, label=leg[ii])
    rect = patches.Rectangle((corner[ii], corner[ii]), len_sect[ii], len_sect[ii], linewidth=2, edgecolor='black', facecolor='none', ls=lst[ii], zorder=100) #, label=leg[ii])
    ax.add_patch(rect)
    
    # get mean acceptance rates
    acc = []
    for jj in range(n_ch):
        stats = functions.load(PurePath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii+5), 'sam_lM_fix\stats'))
        acc.append(stats['out'][jj]['acc'])

    acc_mean = np.mean(np.array(acc)*100, axis=0)
    # acc_std = np.std(np.array(acc)*100, axis=0, ddof=1)
    acc_mean = np.reshape(acc_mean, (len_sect[ii]//q, len_sect[ii]//q), order='C')
    # acc_std = np.reshape(acc_std, (len_sect[ii]//q, len_sect[ii]//q), order='C')

    # write acceptance rate into block
    text_pos = (corner[ii]+q/2, corner[ii]+ii*(q/5+y_tex_pad))
    for jj in range(len_sect[ii]//q):
        for kk in range(len_sect[ii]//q):
            ax.text(text_pos[0]+kk*q + x_tex_corr, text_pos[1]+jj*q + y_tex_corr, f'${acc_mean[jj, kk]:2.1f}$') # \pm{acc_std[jj, kk]:2.0f}')

textstr = (r"$128\times128$"+"\n"+r"$256\times256$"+"\n"+r"$384\times384$"+"\n"+r"$512\times512$")
ax.text(0.85, 0.5, textstr, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black'))
plt.subplots_adjust(right=0.75)

# ax.legend(bbox_to_anchor=(0.13, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=5)
ax.set_axisbelow(True)
ax.set_xlim(0-0.5, 512-0.5)
ax.set_ylim(512-0.5, 0-0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('MLwG block acceptance rates')

plt.tight_layout()
# plt.savefig(plot_dir / 'loc_acc.pdf', dpi=1000)
plt.savefig(plot_dir / 'loc_acc_noCol.pdf', dpi=1000)

#%% Figure 6: computing times

[MALA_WALL, MWG_WALL, MWG_CPU] = computing_times.main()

n_s = 1000 # time per 'n_s' samples
problem = ['512', '384', '256', '128']

MALA_WALL_mean = np.mean(MALA_WALL*n_s, axis=1)[::-1]
MALA_WALL_std = np.std(MALA_WALL*n_s, axis=1)[::-1]
MWG_WALL_mean = np.mean(MWG_WALL*n_s, axis=1)[::-1]
MWG_WALL_std = np.std(MWG_WALL*n_s, axis=1)[::-1]
MWG_CPU_mean = np.mean(MWG_CPU*n_s, axis=1)[::-1]
MWG_CPU_std = np.std(MWG_CPU*n_s, axis=1)[::-1]

SISC(grid=1, fig_height=7, fig_width=10)

plt.figure()

# p = plt.plot(MALA_WALL_mean, label='MALA wall-clock & CPU', zorder=100)
p = plt.plot(MALA_WALL_mean, label='MALA wall-clock & CPU', zorder=100, color='black')
# plt.fill_between(np.arange(4), MALA_WALL_mean-MALA_WALL_std, MALA_WALL_mean+MALA_WALL_std, color=p[-1].get_color(), alpha=0.5, zorder=100)
plt.fill_between(np.arange(4), MALA_WALL_mean-MALA_WALL_std, MALA_WALL_mean+MALA_WALL_std, color=p[-1].get_color(), alpha=0.25, zorder=100)

# p = plt.plot(MWG_WALL_mean, label='MLwG wall-clock', zorder=100)
p = plt.plot(MWG_WALL_mean, label='MLwG wall-clock', zorder=100, color='black', ls=':')
# plt.fill_between(np.arange(4), MWG_WALL_mean-MWG_WALL_std, MWG_WALL_mean+MWG_WALL_std, color=p[-1].get_color(), alpha=0.5, lw=0, zorder=100)
plt.fill_between(np.arange(4), MWG_WALL_mean-MWG_WALL_std, MWG_WALL_mean+MWG_WALL_std, color=p[-1].get_color(), alpha=0.25, lw=0, zorder=100)

# p = plt.plot(MWG_CPU_mean, label='MLwG CPU', zorder=100)
p = plt.plot(MWG_CPU_mean, label='MLwG CPU', zorder=100, color='black', ls='--')
# plt.fill_between(np.arange(4), MWG_CPU_mean-MWG_CPU_std, MWG_CPU_mean+MWG_CPU_std, color=p[-1].get_color(), alpha=0.5, lw=0, zorder=100)
plt.fill_between(np.arange(4), MWG_CPU_mean-MWG_CPU_std, MWG_CPU_mean+MWG_CPU_std, color=p[-1].get_color(), alpha=0.25, lw=0, zorder=100)

plt.xticks([0, 1, 2, 3], [r'$128\times128$'+r' -- $1$', 
                        r'$256\times256$'+r' -- $4$',
                        r'$384\times384$'+r' -- $9$',
                        r'$512\times512$'+r' -- $16$'
                        ])

plt.xlabel('problem size -- number of extra cores for MLwG')
plt.ylabel('time/sample [sec]')

plt.title('Computation times')
plt.legend()

plt.tight_layout()
# plt.savefig(plot_dir / 'comp_times.pdf', dpi=1000)
plt.savefig(plot_dir / 'comp_times_noCol.pdf', dpi=1000)

#%% Figure 7: House: true image, data, mean and CI differences, 512 x 512 image

[x_im, y_im_true, y_im, lam, N, d] = functions.load(data_dir / 'conf12' / 'problem')
par = functions.load(data_dir / 'conf12' / 'par')

stats_path = data_dir / 'conf12' / 'sam_lM' / 'stats'
# stats_path = data_dir / 'conf12' / 'sam_fM' / 'stats'
stats = functions.load( stats_path )

mean = stats['mean']
CI_l = stats['CI'][:, 0]
CI_u = stats['CI'][:, 1]

SISC(grid=0, fig_height=11, fig_width=12)

fig = plt.figure()

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2,2),
                 axes_pad=0.3,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

im = grid[0].imshow(x_im, vmin=0, vmax=par['max_int'], cmap='gray')
grid[0].set_title(r'True image')
grid[0].set_yticks([])
grid[0].set_xticks([])
grid[0].cax.colorbar(im)
grid[0].cax.toggle_label(True)

im = grid[1].imshow(y_im, vmin=0, vmax=par['max_int'], cmap='gray')
grid[1].set_axisbelow(True)
grid[1].set_title(r'Data')
grid[1].set_yticks([])
grid[1].set_xticks([])
grid[1].cax.colorbar(im)
grid[1].cax.toggle_label(True)

im = grid[2].imshow(functions.res(mean, (N,N)), vmin=0, vmax=par['max_int'], cmap='gray')
grid[2].set_title(r'Mean')
grid[2].set_yticks([])
grid[2].set_xticks([])
grid[2].cax.colorbar(im)
grid[2].cax.toggle_label(True)

im = grid[3].imshow(functions.res(CI_u-CI_l, (N,N)), cmap='gray') # , vmin=vmin_CI, vmax=par['max_int']
grid[3].set_title(r'90% CI difference')
grid[3].set_yticks([])
grid[3].set_xticks([])
grid[3].cax.colorbar(im)
grid[3].cax.toggle_label(True)

plt.tight_layout()
plt.savefig(plot_dir / 'house.pdf', dpi=1000)


plt.show()
