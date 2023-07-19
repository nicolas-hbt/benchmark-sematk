import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colourWheel =['#77DD77',
            '#ff6961',
            '#779ECB',
            '#966FD6',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = ['solid', 'dashed', 'dashdot', 'dotted']
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

def add_floating_axis1(ax1):
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 30)
    axis.label.set_text(r"$\theta = 30^{\circ}$")
    axis.label.set_visible(True)

    return axis

def add_floating_axis2(ax1):
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")
    axis.label.set_visible(True)

    return axis

def plot_mrr_sem(model, dataset, which_sem='ext', start_y=0.5, spacing_btw_epochs=6):
    '''
    which_sem = {CWA, WUP, OWA, ext}
    '''
    plt.rcParams["figure.figsize"] = [17, 10]
    plt.rcParams["figure.autolayout"] = True
    
    if dataset == 'Codex-S':
        max_mrr = 0.5
    elif dataset == 'Codex-M':
        max_mrr = 0.4
    elif dataset == 'WN18RR':
        max_mrr = 0.5
    elif dataset == 'YAGO37K':
        max_mrr = 0.5
    elif dataset == 'DB93K':
        max_mrr = 0.4
    elif dataset == 'FB15K237':
        max_mrr = 0.4
    elif dataset == 'YAGO4-18K':
        max_mrr = 1.0
    table = pd.read_csv('models/'+dataset+'/'+model+'.csv', sep=",", index_col=0)
    table = table.T.iloc[0:]

    try:
        max_ep, min_ep = max(list((table.index).map(int))), min(list((table.index).map(int)))
    except:
        table = pd.read_csv('models/'+dataset+'/'+model+'.csv', sep=";", index_col=0)
        table = table.T.iloc[0:]
        max_ep, min_ep = max(list((table.index).map(int))), min(list((table.index).map(int)))
    ep_step = int(max_ep/len(table.index))
    mrr = list(table['MRR'].values)
    if which_sem == 'CWA':
        s1 = list(table['CWA_Sem@1'].values)
        s3 = list(table['CWA_Sem@3'].values)
        s10 = list(table['CWA_Sem@10'].values)
    elif which_sem == 'WUP':
        s1 = list(table['WUP_Sem@1'].values)
        s3 = list(table['WUP_Sem@3'].values)
        s10 = list(table['WUP_Sem@10'].values)
    elif which_sem == 'ext':
        s1 = list(table['Sem@1'].values)
        s3 = list(table['Sem@3'].values)
        s10 = list(table['Sem@10'].values)
    s1 = [_/100 for _ in s1]
    s3 = [_/100 for _ in s3]
    s10 = [_/100 for _ in s10]
        
    fig, ax1 = plt.subplots()
    ax1.set_alpha(0.01)

    s1 = ax1.plot(table.index, s1, color=colourWheel[1], linestyle=dashesStyles[1], lw=3, alpha=0.7)
    s3 = ax1.plot(table.index, s3, color=colourWheel[2], linestyle=dashesStyles[2], lw=3, alpha=0.7)
    s10 = ax1.plot(table.index, s10, color=colourWheel[3], linestyle=dashesStyles[3], lw=3, alpha=0.7)
    ax2 = ax1.twinx()
    mrr = ax2.plot(table.index, mrr, color=colourWheel[0], linestyle=dashesStyles[0], lw=3, alpha=0.7)
    
    ax1.set_ylim([start_y, 1])

    ax2.set_ylim([0.0, max_mrr])
    ax2.set_xlim([0, len(table.index)-1])
    
    ax1.set_xticks(np.arange(0, len(table.index)+1, spacing_btw_epochs))
    ax1.tick_params(axis='both', which='major', pad=17.5, labelsize=30)
    ax2.tick_params(axis='both', which='major', pad=17.5, labelsize=30)
    ax1.set_xlabel('Epochs', labelpad=15, fontsize=35)
    ax1.set_ylabel('Sem@K', labelpad=15, fontsize=35)
    ax2.set_ylabel('MRR', color='black', labelpad=25, fontsize=35)
    
    plt.savefig(dataset+'-'+model+'.pdf')

# Function calls
for model in ["TransE", "TransH", "DistMult", "ComplEx", "SimplE", "ConvE", "ConvKB", "RGCN", "CompGCN"]:
    for dataset in ["YAGO4-18K", "YAGO37K", "FB15K237", "DB93K"]:
        plot_mrr_sem(model, dataset, which_sem='CWA', start_y=0.5, spacing_btw_epochs=6)
    for dataset in ["Codex-S", "Codex-M", "WN18RR"]:
        plot_mrr_sem(model, dataset, which_sem='ext', start_y=0.5, spacing_btw_epochs=6)
