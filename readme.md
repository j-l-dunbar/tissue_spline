# tissue_spline Tool

## tissue_spline_Driver.py
Processes raw coordinate data from sections of retinal tissue, and performs a non-linear principal component analysis to estimate the natural tissue axis. A pretrained autoencoder neural network is used to estimate the retinal tissue axis. Residuals from the projection are used to define an orthogonal axis to the primary tissue axis. Retinal tissue layers are estimated in the linearized data using k-means clustering. 

## tissue_spline_Utils.py
### read_imaris_csv()
Takes path to csv of imaris output. Reads and renames relevant columns. 
- list_cols - csv columns to keep
- rename_cols - new names for kept columns from csv
- skiprows - skips the first N rows of the csv before the column headers


### plot_change()
shows the change in coordinates before and after processing the data
- before: raw X,Y coordinates
- after: processed X,Y coordinates


### distance_to_nth_neighbor()    
Finds the distances to each point's n-th nearest neighbor
- nth_neighbor: n denoting n-th nearest neighbor


### filter_outliers()
Filters dataframe based on distances to n-nearest nighbours   

### normalize()
Returns data normalized between -1 and 1 
- reference - if provided, function will standardize data using max and min values from the reference


### rescale_data()
Returns the dataframe with two new columns with normalized X and Y columns, preserving the aspect ratio   


### reference_channel()
Returns a dataframe filtered to contain only a specified reference object (usually DAPI)
- nth_neighbor - optionally filters on distance to nth-nearest neighbor using max_dist


### make_model()
Conditionally makes or loads an Autoencoder Neural Network to reduce the data onto 1 dimension 
- verbose - show results during training -- 0: nothing, 1: everything, 2: epoch result, 3: epoch number
- fresh_model -- if true, will construct a completely new model
- save -- if true, will (over)write the model to be reused


### autoencoder_model()
Returns an Autoencoder model to determine a Pincipal Curve through the data
- dims - number of dimensions to the data (only tested with 2D so far -- Xnorm, Ynorm)
- nodes - number of neurons in the first densly connected layer -- more neurons allows for more complex curves to be traced
- Framework inspired by prinpy.glob.NLPCA()


### sort_using_model()
Accesses the hidden bottle neck layer and uses the values to sort the data in order by the reduced dimension   


### trace_spline_on_predictions()
Takes the coordinate data representing the autoencoder's dimension reduction and traces a spline.
Takes the coordinate data representing the autoencoder's dimension reduction and traces a spline.
This is done because the bottle-neck layer is non-linear, and the tissue axis cannot be accurately represented without this step


### process_df()
Updates each row of the dataframe to include its new position in the dimension reduction. 
- Calculates the residuals of this projection. 


### closest_one()
finds the closes point on the spline for each datum. 
- returns the index (i.e., the 1D position) and distance of projection.


### ave_dist()
calculates the (Euclidian) L2 Norm of projection on the spline.   
calculates the (Euclidian) L2 Norm of projection on the spline.   


### kmeans_clusters()
Takes 1D data and finds clusters
- i: the 1D data (intended to be used to cluster the residual data)
- algorithm: strategy used for clustering
- init: the initial 'seed' points for the clusters (set according to the expected profile of the data)    


### calc_clusters()
Takes the processed dataframe and performs clustering on the DAPI points, to determing the GCL, INL and ONL of the retina   




