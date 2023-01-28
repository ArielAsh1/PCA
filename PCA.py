# implementation of the PCA algorithm on the MNIST dataset

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load the MNIST dataset from sklearn datasets.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# the PCA algo
class PCA:
    def __init__(self):
        self.V = None

    def fit(self, X):
      
        # make the mean of the columns of the dataset to 0.
        X = X - X.mean()
        # calculate the covariance matrix
        C = np.cov(X, rowvar=False)
        # calculate the eigenvalues and eigenvectors of the covariance matrix C
        eig_vals, eig_vecs = np.linalg.eig(C)
        # get highest to lowest eig_vals
        indices = np.argsort(eig_vals)[::-1]
        # sort eig_vecs according to eig_vals from highest to lowest
        self.V = eig_vecs[:, indices]

        return self.V

    def transform(self, X, n_dimentions):
        V = self.V[:, :n_dimentions]
        X_reduced = np.dot(V.T, X.T)

        return X_reduced

# Run the algorithm on MNIST dataset
pca = PCA()
V = pca.fit(X)
# transform the data to the top 2 principal components
X_reduced = pca.transform(X, 2).T

# Plot the data in the top 2 principal component space
y = [int(label) for label in y]
fig, ax = plt.subplots()
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')

cbar = fig.colorbar(scatter)
cbar.set_label('Labels')

ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
plt.show()


"""
V in {R}^{d\times r}$ is the matrix whose colmns are the top r eigenvectors of X^TX.
That is, the eigenvectors that correspond to the r largest eigenvalues.
"""
#plots the two matrices V^TV and VV^T
def plot_matrix(M, m_name):
    fig, ax = plt.subplots()
    plt.title(f"Marix ${m_name}$", fontdict=font)
    ax.set_xlabel('Columns', fontdict=font2)
    ax.set_ylabel('Rows', fontdict=font2)
    plt.imshow(np.real(M))

r = 2
V_reduced = V[:, :r]

# set fonts to plot
font = {'family': 'serif', 'color': 'black', 'size': 15}
font2 = {'family': 'serif', 'color': 'black', 'size': 12}

# plot metrics
plot_matrix(np.dot(V_reduced.T, V_reduced), "V^T*V")  # plot V_reduced.T * V_reduced
plot_matrix(np.dot(V_reduced, V_reduced.T), "V*V^T")  # plot V_reduced * V_reduced.T

# projects a sample to a n - dimentional space and recontsturct it to the original space.
def pca_reconstruction(x, n_dimentions):
    V_reduced = V[:, :n_dimentions]
    x_reconstructed = np.dot(V_reduced, x)
    return x_reconstructed


# samples a random image from the dataset
# and uses the function above to project it into n-dimensional space and reconstruct it to the original space.
# the reconstructed image IS NOT the original image.
# the reconstruction is from spaces of dimensions: 3, 10, 100

head_font = {'family':'serif','color':'black','size':16}
# Get a random X_i image and plot it
images = np.random.choice(range(64),3)  # three random indices
X_i = X[images[0]]
X_i = X_i[:, np.newaxis].T
Xi_image = X_i.reshape(28,28)
plt.title("Original Image", fontdict=head_font)
plt.imshow(Xi_image.squeeze(), cmap='Greys_r')
plt.show()

# dimensions space to project X_i
dimensions = [3, 10, 100]

# use PCA to find V to project X_i to dim from dimensions
pca = PCA()
V = pca.fit(X)

for dim in dimensions:
    # transform Xi to a d dimension components
    Xi_reduced = pca.transform(X_i, dim)  # X_i_reduced.shape: (1, d)
    # reconstruct Xi
    Xi_reconstructed = pca_reconstruction(Xi_reduced, dim)  # X_i_reconstruct.shape: (1, 784)

    # now print the reconstructed image
    Xi_image = Xi_reconstructed.reshape(28, 28)
    plt.title(f"Reconstructed from {dim} dimensions", fontdict=head_font)
    plt.imshow(Xi_image.squeeze(), cmap='Greys_r')
    plt.show()
    
