import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from scipy import interpolate
from scipy.spatial import distance

import tensorflow.keras as k
k.backend.set_floatx('float64') # ****** fixes warning 
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

# read in coordinate data
def read_imaris_csv(file_path, skiprows=3, list_cols=['Position X', 'Position Y', 'Unit','Surpass Object', 'ID'], rename_cols=['X', 'Y', 'Unit', 'Object', 'ID']):
    """ Takes path to csv of imaris output. Reads and renames relevant columns. 
        list_cols - csv columns to keep
        rename_cols - new names for kept columns from csv
        skiprows - skips the first N rows of the csv before the column headers
    """
    print(f'Reading file "{file_path}"')
    data = pd.read_csv(file_path, skiprows=range(skiprows))
    data = data[list_cols]
    data.columns = rename_cols
    return data.copy()

# visualize data
def plot_change(before, after, size=0.1):
    """ shows the change in coordinates before and after processing the data
        before: raw X,Y coordinates
        after: processed X,Y coordinates
    """
    print("Plotting Change...")
    fig, ax = plt.subplots()
    ax.scatter(*before, s=size, alpha=0.7)
    ax.scatter(*after, s=size)
    ax.set_aspect('equal')
    return fig

# cleanup data
def distance_to_nth_neighbor(x, y, nth_neighbor): # TODO could I do a mean distance of the n-nearest nighbours for each point?
    """ Finds the distances to each point's n-th nearest neighbor
        nth_neighbor: n denoting n-th nearest neighbor
    """
    point_data = np.column_stack((x, y)) # joins the X and Y data
    nbrs = NearestNeighbors(n_neighbors=nth_neighbor+1, algorithm='ball_tree').fit(point_data)
    distances, _ = nbrs.kneighbors(point_data) # find the distances to the n-closest neighbors
    return distances.T[nth_neighbor] # transpose, and get n-th row (first row is just idenity)

def filter_outliers(df, nth_neighbor, max_dist, show=True):
    """ Filters dataframe based on distances to n-nearest nighbours """
    print(f'Calculating distances to nearest neighbor {nth_neighbor} (max: {max_dist})')
    filtered = df[distance_to_nth_neighbor(df.X, df.Y, nth_neighbor) <= max_dist].copy()
    if show: # plots the df before and after filtering
        plot_change([df.X, df.Y], [filtered.X, filtered.Y], size=1)
        plt.show()
    return filtered

def normalize(data, reference=None):
    """ Returns data normalized between -1 and 1 
        reference - if provided, function will standardize data using max and min values from the reference
    """
    if reference==None: reference = data
    return (data - reference.min()) / (reference.max() - reference.min()) * 2 - 1 

def rescale_data(df):
    """ Returns the dataframe with two new columns with normalized X and Y columns, preserving the aspect ratio """
    aspect_ratio = (df.Y.max() - df.Y.min()) / (df.X.max() - df.X.min())
    print("Normalizing data to (-1, 1)")
    print(f"Aspect Ratio: {np.around(aspect_ratio, 4)}")
    df['Xnorm'] = normalize(df.X)
    df['Ynorm'] = normalize(df.Y) * aspect_ratio # preserves the aspect ratio for the (x,y) data
    return df


# project data into 1D using Autoencoder Neural Network
def reference_channel(df, ref_obj, nth_neighbor=None, max_dist=0.1, show=True):
    """ Returns a dataframe filtered to contain only a specified reference object (usually DAPI)
        nth_neighbor - optionally filters on distance to nth-nearest neighbor using max_dist
    """
    ref_df =  df[df.Object==ref_obj].copy() # uses only one reference object
    if nth_neighbor==None: pass # skips distance filtering
    else: 
        print(f"Calculating {ref_obj} Nearest neighbor {nth_neighbor} (max: {max_dist})")
        ref_df = ref_df[distance_to_nth_neighbor(ref_df.Xnorm, ref_df.Ynorm, nth_neighbor) <= max_dist].copy() # filters the df based on distance to the nth neighbor
    if show: # plots df before and after filtering
        plot_change([df.Xnorm, df.Ynorm], [ref_df.Xnorm, ref_df.Ynorm])
        plt.show()
    return ref_df


def make_model(df, epochs=None, verbose=2, batch_size=50, fresh_model=False, save=False):
    """ Conditionally makes or loads an Autoencoder Neural Network to reduce the data onto 1 dimension 
        verbose - show results during training -- 0: nothing, 1: everything, 2: epoch result, 3: epoch number
        fresh_model -- if true, will construct a completely new model
        save -- if true, will (over)write the model to be reused
    """
    data=df[['Xnorm', 'Ynorm']]
    if fresh_model==True: # makes a model from scratch
        model = autoencoder_model(dims=2, nodes=40)
    else: 
        model = km.load_model('deep_spline') # uses the pretrained model (better because L is always 0 and R is always 1, axis doesn't collapse onto 0)
    if epochs: 
        model.fit(data, data, epochs=epochs, verbose=verbose, batch_size=batch_size) # trains the model further on the input data
        plt.plot(model.history.history['loss'])
        plt.show()
    if save==True: model.save('deep_spline') # if save true, model will be used to write over the saved "retinal_spline_model" folder
    return model, np.array(data)


def autoencoder_model(dims=2, nodes=60):
    """ Returns an Autoencoder model to determine a Pincipal Curve through the data
        dims - number of dimensions to the data (only tested with 2D so far -- Xnorm, Ynorm)
        nodes - number of neurons in the first densly connected layer -- more neurons allows for more complex curves to be traced
            Framework inspired by prinpy.glob.NLPCA()
    """
    optimizer = k.optimizers.Adam(learning_rate=0.0025) # decides how the model is improved at each step
    model = km.Sequential(name='AutoEncoder_Model') # initializes the model type (sequential neural network)
    model.add(kl.Dense(dims, name='input_1')) # takes in the (2D) point data
    model.add(kl.Dense(nodes, activation='softmax', name='Dense_layer1')) # morphs the data to be linearly interpolated
    model.add(kl.Dense(nodes, activation='tanh', name='Dense_layer11'))
    # model.add(kl.Dropout(0.01))
    model.add(kl.Dense(nodes, activation='tanh', name='Dense_layer111'))
    model.add(kl.Dense(1, activation='softplus', name='bottle'))    # bottleneck layer to force a 1D representation of the data
    model.add(kl.Dense(nodes, activation='tanh', name='Dense_layer222'))
    # model.add(kl.Dropout(0.01))
    model.add(kl.Dense(nodes, activation='tanh', name='Dense_layer22'))
    model.add(kl.Dense(nodes, activation='softmax', name='Dense_layer2')) # projects the 1D data back into 2D (in the style of an AutoEncoder)
    model.add(kl.Dense(dims, name="output_1")) # returns the final projected data 
    model.compile(loss='mean_squared_error', optimizer=optimizer) # minimizes the mean-squared 'distance' between the inputs and the outputs of the model
    return model

def sort_using_model(model, data, show=True):
    """ Accesses the hidden bottle neck layer and uses the values to sort the data in order by the reduced dimension """
    intermediate_model = k.Model(inputs=model.input, outputs=model.get_layer('bottle').output) # takes the values from the bottleneck layer to use for sorting the dataframe 
    predictions = model.predict(data) 
    pred_order = intermediate_model.predict(data) # gets the 1D data from the trained model
    all = np.concatenate([predictions, pred_order], axis = 1) # used to order the indices of the dataframe based on their bottleneck position
    all_sorted = all[all[:,2].argsort()] # sorts the output data based on the prediction order
    predictions = all_sorted[:,0:2] # gets the relevant columns to return as a list of points representing the projected 1D data
    if show: 
        plot_change(data.T, predictions.T)
        # plt.show()
    return predictions

def trace_spline_on_predictions(df, predictions, jump_every=5, precision=4, show=True):
    """ Takes the coordinate data representing the autoencoder's dimension reduction and traces a spline it
        This is done because the bottle-neck layer is non-linear, and the tissue axis cannot be accurately represented without this step
    """
    print(f"Tracing spline with {precision} digit precision.")
    spl_targets = predictions   # array of coodinates from the spline estimate
    data_points = np.dstack((df.Xnorm, df.Ynorm))[0]    # array of normalized coordinates from input data (stacks them into an array of coordinates)
    Xs = np.append(spl_targets.T[0,::jump_every], spl_targets.T[0,-1])  # skip every nth entry in array (solves the problem of imperfect ordering in the bottle neck layer)
    Ys = np.append(spl_targets.T[1,::jump_every], spl_targets.T[1,-1])
    tck, _ = interpolate.splprep([Xs, Ys], s=0.001) # "connect the dots" (takes the points representing the spline and draws a line through them) -- no smoothing when s=0
    unew = np.arange(0,1,1/(10**precision)) # makes N points to represent the interpolated spline
    spline_coords = np.array(interpolate.splev(unew,tck))   # array of evenly spaced points representing the interpolated spline
    if show:
        plot_change(spl_targets.T, spline_coords)
        plt.show()
    return spline_coords, data_points

# use spline to update the dataframe
def process_df(df, spline_coords, data_points, max_dist=0.4):
    """ Updates each row of the dataframe to include its new position in the dimension reduction. 
        Calculates the residuals of this projection 
    """
    df['index_close'], df["dist_proj"] = closest_one(data_points, spline_coords.T)  # for each datum, finds the closest point on the predicted spline
    
    precision = spline_coords.shape[1] # gets the number of points that represent the spline, which represents the precision of the projection to 1D
    df['norm1D'] = df.index_close/(precision)  # number between 0 and 1 representing the nearest position along the spline
    df['Xproj'] = spline_coords[0][df.index_close]  # x coord of the projected value onto the spline
    df['Yproj'] = spline_coords[1][df.index_close]  # y value of projection

    df["radius"] = np.sqrt(df["Xnorm"]**2 + df["Ynorm"]**2) # distance to the origin
    df["radius_proj"] = np.sqrt(df["Xproj"]**2 + df["Yproj"]**2)    # how far was the projection onto the spline
    df["dist_ori"] = (df["radius"] - df["radius_proj"]) # if the point moved towards or away from the origin (which side of the spline was the data originally on?)
    df["residuals_proj"] = -(df["dist_ori"]/abs(df.dist_ori)) * df["dist_proj"] # distance of projection with the sign based on change relative to the origin during projection
    df = df[(df.index_close.between(df.index_close.min()+1, df.index_close.max()-1)) &
            (df.residuals_proj < max_dist)].copy()
    return df

def closest_one(points, targets):
    """finds the closes point on the spline for each datum 
        returns the index (i.e., the 1D position) and distance of projection
    """
    distances = distance.cdist(points, targets) # measures all the distances between points of each array
    min_args = np.argmin(distances, axis = 1)   # finds the index of the smallest distance value for each row in the array
    min_dists = np.amin(distances, axis = 1)    # finds the smallest distance value for each row in the array
    return min_args, min_dists

def ave_dist(points):
    """ calculates the mean (Euclidian) L2 Norm of projection on the spline """
    point_array = np.array(points[['residuals_proj', 'norm1D']])
    point_ref = np.array(points[points.Object=='DAPI'][['residuals_proj', 'norm1D']])
    distances = distance.cdist(point_array, point_array)
    return distances.mean(axis=0)/distances.std(axis=0)


# Use the new dataframe to meaningfully cluster the data into retinal layers
def kmeans_clusters(i, cluster_seeds, algorithm='elkan'):
    """ Takes 1D data and finds clusters
        i: the 1D data (intended to be used to cluster the residual data)
        algorithm: strategy used for clustering
        init: the initial 'seed' points for the clusters (set according to the expected profile of the data)    
    """
    n_clusters = 4
    c1, c2, c3, c4 = cluster_seeds # scales the min and max values to represent the nuclear layers in terms of their residual values from projection
    init = [i.nsmallest(5).max()*c1, i.nsmallest(5).max()*c2, i.nlargest(5).min()*c3, i.nlargest(5).min()*c4]
    k_means = KMeans(n_clusters=n_clusters, algorithm=algorithm, n_init=1, init=np.array(init).reshape(-1,1)) # 1D kmeans clustering of the data
    return k_means.fit_predict(np.array(i).reshape(-1,1))


def calc_clusters(df, n_groups=6, cluster_seeds = [0.9, 0.01, 1.15, 1.2], max_proj=0.12, nth_isolated = 300):
    """ Takes the processed dataframe and performs clustering on the DAPI points, to determing the GCL, INL and ONL of the retina """
    # df = df[df.residuals_proj<max_proj].copy()
    df.sort_values('norm1D', inplace=True)
    df['process_group'] = df.norm1D*n_groups//1 # cuts the 1D axis into n groups for clustering (works better than trying to cluster the whole axis all at once)
    df['group_distance'] = np.concatenate(df.groupby('process_group').apply(lambda i: ave_dist(i)))
    df = df[df.group_distance < df.group_distance.nlargest(nth_isolated).iloc[-1]]
    
    df_process_group = df[df.Object =='DAPI'].groupby('process_group') # splits dataframe into chunks to process separately into kmeans_clusters (works better than taking the whole thing all together)
    df['clusters'] = 'NaN' # creates an empty column to be filled below
    
    # finds the clusters from the residual projection distance -- scaled values for seeding the kmeans clustering (corresponds to the distances for the ONL(out), ONL(in), INL, and GCL, respectively) (average of the 5 largest or smalles values)
    df.loc[df.Object =='DAPI', 'clusters'] = df_process_group['residuals_proj'].transform(lambda i: kmeans_clusters(i, cluster_seeds)) 
    df_DAPI = df[df.Object =='DAPI'].reset_index() # creates a new df with only values from DAPI to be used to as a reference to assigning layers for the remaining non-DAPI points (DAPI clusters set the values for the other objects)
    df['closest_DAPI'] = closest_one(points=df[['Xnorm', 'Ynorm']], targets=df_DAPI[['Xnorm', 'Ynorm']])[0] # finds the closest DAPI point to each data point
    df['clusters'] = df['closest_DAPI'].apply(lambda x: df_DAPI['clusters'].iloc[x])  # assigns the cluster value from the nearest DAPI point (corresponding to its retinal layer)
    return df

