import matplotlib.pyplot as plt

def SISC(grid=1, fig_width=7.5, fig_height=5): # max width 13

    cm = 1/2.54
    plt.rcdefaults()

    rcParams = {\
    'lines.linewidth':.7,\
    'lines.markersize':1.9,\
    'legend.fontsize':7,\
    'legend.fancybox':False,\
    'xtick.labelsize':6,\
    'ytick.labelsize':6,\
    'text.usetex':False,\
    'font.family':'serif',\
    # 'font.name': 'Computer Modern Roman',\
    'mathtext.fontset':'cm',\
    # 'mathtext':'regular',\
    'figure.figsize':(fig_width*cm, fig_height*cm),\
    'font.size':7,\
    'axes.prop_cycle': plt.cycler(color=plt.cm.tab10.colors)
    }

    if grid:
        rcParams.update({
            'axes.grid':True,
            'grid.linewidth':.3,
            'grid.color':'lightgrey'
            })

    plt.rcParams.update(rcParams)