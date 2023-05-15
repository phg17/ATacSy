#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:53:40 2022

@author: phg17
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def scabox(data, labels, ylabel = 'Syllable Recognition (%)',
           positions = [],
           yticks = [0.5,0.6,0.7,0.8],
           boxwidths = 0.5, size_box = 2.5,
           size_scatter = 25, size_fig = [7,5],
           size_ticks = 16, size_label = 18):
    fig,ax = plt.subplots()
    c = 'black'
    if len(positions) == 0:
        positions = np.arange(1,data.shape[1]+1)
    
    ax.boxplot(data,widths=boxwidths,patch_artist=True,
               positions= positions,
                boxprops=dict(facecolor="white", color=c,linewidth = size_box),
                flierprops=dict(color="white", markeredgecolor=c,linewidth = size_box),
                medianprops=dict(color=c,linewidth = size_box),
                whiskerprops=dict(color=c,linewidth = size_box),
                capprops=dict(color=c,linewidth = size_box),
                meanprops = dict(linestyle='--', linewidth=size_box, color='k')
                )
    ax.scatter(positions,np.mean(data,axis=0),c = 'k',alpha = 1,zorder=5,s=size_scatter*2)
    
    for i in data:
        ax.scatter(positions,i,c = 'grey',alpha = 1,zorder=4,s=size_scatter*2,edgecolor = 'k')
    
    ax.set_xticklabels(labels, size = size_ticks)
    ax.set_ylabel(ylabel, size = size_label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels = (np.asarray(yticks)*100).astype(int), size = size_label)
    fig.set_size_inches(size_fig)
    fig.tight_layout()
    
    return fig, ax
    
    
def plot_significance(ax,x,y, text, text_pos = [0.5,0.5],color = 'k', linewidth = 1.5, fontsize = 15):
    ax.plot([x[0],x[0]],[y[0],y[1]],color = color,linewidth = linewidth)
    ax.plot([x[1],x[1]],[y[0],y[1]],color = color,linewidth = linewidth)
    ax.plot([x[0],x[1]],[y[1],y[1]],color = color,linewidth = linewidth)
    ax.text((x[0]+x[1]) * text_pos[0],y[1] + (y[1] - y[0]) * text_pos[1], text,color = color, fontsize = 15, weight = 'bold')
    

def make_rectangle(ax,x,y,width, height, alpha = 1, facecolor = 'k', edgecolor = 'k',
                   linewidth = 2, zorder = 0):
    rectangle = [Rectangle((x, y), width, height)]
    pc = PatchCollection(rectangle, facecolor = facecolor, edgecolor = edgecolor,
                         alpha = alpha, linewidth = linewidth, zorder = zorder)
    ax.add_collection(pc)
    
    return ax

def plot_cluster(reshaped_pval, pval_lim_color = 0.05, vmin = -1, vmax = 3):
    
    reshaped_log = np.max(-np.log10(reshaped_pval),axis=2)
    reshaped_log_plot = np.nan * np.ones_like(reshaped_log)
    for i in range(reshaped_log_plot.shape[0]):
        for j in range(reshaped_log_plot.shape[1]):
            if reshaped_log[i,j] > -np.log10(pval_lim_color):
                reshaped_log_plot[i,j] = reshaped_log[i,j]

    
    fig, ax = plt.subplots()
    im = ax.imshow(reshaped_log, cmap = plt.cm.gray,
                   aspect = 'auto', origin = 'lower',vmin = -1, vmax = 3)
    im = ax.imshow(reshaped_log_plot, cmap = plt.cm.Reds,
                   aspect = 'auto', origin = 'lower',vmin = -np.log10(0.05), vmax = 3)
    
    return fig, ax, im

def plot_cluster_rawpval(reshaped_pval, pval_lim_color = 0.05, vmin = 0.0, vmax = 0.05):
    
    reshaped = np.min((reshaped_pval),axis=2)
    reshaped_plot = np.nan *np.ones_like(reshaped)
    for i in range(reshaped_plot.shape[0]):
        for j in range(reshaped_plot.shape[1]):
            if reshaped[i,j] < pval_lim_color:
                reshaped_plot[i,j] = reshaped[i,j]

    
    fig, ax = plt.subplots()
    im = ax.imshow(-reshaped, cmap = plt.cm.gray,
                   aspect = 'auto', origin = 'lower',vmin = -1, vmax = 0.0)
    im = ax.imshow(-reshaped_plot, cmap = plt.cm.Reds,
                   aspect = 'auto', origin = 'lower',vmin = -0.05, vmax = 0.0)
    
    return fig, ax, im