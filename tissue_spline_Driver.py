import sys
sys.path.append('.')
import os
from tissue_spline_Utils import *

coordinates_path = r"ZEN_ScanRegion_Detailed.csv"
data_in = read_imaris_csv(coordinates_path, skiprows=3)

# removes spurious data, well away from the main tissue axis
df = filter_outliers(data_in, nth_neighbor=30, max_dist=150, show=True)
df = rescale_data(df) # normalizes the coordinate axes

# sets the coordinate data to be used to refine the 
ref_df = reference_channel(df, 'DAPI', nth_neighbor=15, max_dist=0.01, show=True)

# loads a pre-trained model, to be refined for the raw retinal coordinate data
model, data = make_model(ref_df, epochs=40, verbose=2, fresh_model=False)
predictions = sort_using_model(model, data, show=True)
spline_coords, data_points = trace_spline_on_predictions(df, predictions, show=True)

# combines the calculated spline with the raw coordinate data for further processing
df_annot = process_df(df, spline_coords, data_points)

temp_df = df_annot.copy()

#defines the seeds that will be used to claculate the 1D clusters in the data, representing reintal layers
df_filt = calc_clusters(temp_df, n_groups=24, cluster_seeds = [0.9, 0.0, 1.2, 1.1], max_proj=0.12, nth_isolated=100) 

# removes coordinates that are too far away from the tissue axis
df_outliers = df_filt[df_filt.residuals_proj > 0.12]

fig, ax = plt.subplots(figsize=(20, 5))
sns.scatterplot(x='norm1D', y='residuals_proj', data=df_filt[df_filt.Object=='DAPI'], hue='clusters', s=3, ax=ax)

def scatter_plots(df_filt, predictions, plotting_group):
    """ Makes a Scatterplot of the clustered data in 'image coordinates' (top) and in 'retinal coordinates' (bottom) """
    sns.set_context('talk') # makes the graph labels larger and easier to read
    fig, ax = plt.subplots(nrows=2, figsize=(26, 20), gridspec_kw={'height_ratios': [4.5, 1]}) # sets the height and width for each of the subplots in the figure
    sns.scatterplot(x='Xnorm', y=-df_filt['Ynorm'], hue=df_filt.clusters, data=df_filt, palette='deep', s=15, ax=ax[0]) # creates the 'image coordinates' scatterplot
    ax[0].plot(predictions.T[0], -predictions.T[1], linewidth=1, c='k') # plots the 
    ax[0].legend(['Retinal Axis', 'GCL','ONL','MidNL', 'INL'], loc='lower right')
    sns.scatterplot(data=df_filt, x='norm1D', y='residuals_proj', hue=df_filt['clusters'], palette='deep', s=15, ax=ax[1])
    ax[1].axhline(0, c='k')
    ax[1].legend(['']).remove()
    fig.suptitle(plotting_group)
    return fig

fig_scatter = scatter_plots(df_filt[(df_filt.Object == 'DAPI')], predictions=spline_coords.T, plotting_group="DAPI")

def joint_plots(coordinates, hue, plotting_group):
    """Makes a scatter plot of the 1D index data plotted against the estimated residuals of projection. 
    Histograms of each parameter are also plotted on each respective axis"""
    joint = sns.jointplot(x = coordinates.norm1D, y = coordinates["residuals_proj"], hue = coordinates[hue], palette = "deep", legend=[], xlim=[0,1], s=15)
    # joint.plot_marginals(sns.histplot, kde=True) # changes the marginal KDE plots to histograms
    joint.fig.set_figwidth(28) # adjusts the size of the plot
    joint.fig.set_figheight(7)                                                              
    joint.ax_joint.axhline(y = coordinates["residuals_proj"].mean(), color = 'r')
    joint.ax_joint.axhline(y = 0, color='k') # Marks the zero axis, representing the spline running though the data
    joint.ax_marg_y.axhline(y = 0, color='k') # Marks the zero line for the Y-axis histplot
    # joint.ax_marg_x.axhline(y = 1/(max(coordinates["norm1D"])), color='k') # Marks a uniform distribution for the given number of points
    joint.fig.suptitle(plotting_group)
    return joint.fig

def hist_plots(df_filt, plotting_group, n_bins=25, cluster=2, xlabel='GCL'):
    """ Makes a histplot of the data in the specificed layer """
    fig, ax = plt.subplots()
    sns.set_context('talk')
    sns.histplot(df_filt[(df_filt['clusters']==cluster)]['norm1D'], bins=n_bins, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_title(plotting_group)
    fig.tight_layout()
    return fig

plot_objects = sorted(list(set(df_filt.Object)))

path = '?' # path where the files will be saved
fname = '?' # details of the coordinate data processed

path_plots = path+'plots/'

if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

# saves the various plots
for plotting_group in plot_objects:
    joint_fig = joint_plots(df_filt[df_filt.Object == plotting_group], hue='clusters', plotting_group=plotting_group)
    joint_fig.savefig(path_plots+plotting_group+'_residual.pdf')

    fig_scatter = scatter_plots(df_filt[df_filt.Object == plotting_group], predictions=spline_coords.T, plotting_group=plotting_group)
    fig_scatter.savefig(path_plots+plotting_group+'_scatter.pdf')

    fig_hist = hist_plots(df_filt[df_filt.Object == plotting_group], plotting_group, cluster=3)
    fig_hist.savefig(path_plots+plotting_group+'_hist.pdf')


# saves the relevant spline information from the processed CSV
df_final = df_filt[['Object', 'X', 'Y', 'Xnorm', 'Ynorm',
                    'decimal1D', 'Xproj', 'Yproj', 
                    'residuals_proj', 'norm1D', 'clusters']]
df_final.to_csv(path+'Processed_'+fname) 
