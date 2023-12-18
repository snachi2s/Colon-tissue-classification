'''
Visualizing the feature space in 2D and 3D
'''

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

#REF: https://plotly.com/python/subplots/

features_dataframe = pd.read_csv('hist_features_dataframe.csv')
X = features_dataframe.drop(['image_id','label'], axis=1)
X = features_dataframe.drop(['image_id','label','contrast1','contrast2','contrast3','contrast4','dissimilarity1','dissimilarity2','dissimilarity3','dissimilarity4','homogeneity1','homogeneity2','homogeneity3','homogeneity4','energy1','energy2','energy3','energy4','correlation1','correlation2','correlation3','correlation4'], axis=1)
y = features_dataframe['label']

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(X)

#combined feature space visualization(using plotly)
# fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], title = 'All class visualization', 
#                  color=y, labels={'Label':'Class'}, hover_name=y)
# fig.show()

unique_classes = np.unique(y)
n_rows = 2
n_cols = 2

plt.figure(figsize=(15, n_rows * 5))

for i, class_label in enumerate(unique_classes):
    plt.subplot(n_rows, n_cols, i + 1)
    indices = y == class_label
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1])
    plt.title(f'Class {class_label}')

plt.tight_layout()
plt.show()

#classwise visualization using plotly 

# fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=("Class 0", "Class 1", "Class 2", "Class 3"))

# indices = np.where(y == unique_classes[0])
# fig.add_trace(go.Scatter(x=tsne_results[indices, 0].flatten(), y=tsne_results[indices, 1].flatten(), mode='markers', name=f'Class {unique_classes[0]}'), row=1, col=1)

# indices = np.where(y == unique_classes[1])
# fig.add_trace(go.Scatter(x=tsne_results[indices, 0].flatten(), y=tsne_results[indices, 1].flatten(), mode='markers', name=f'Class {unique_classes[1]}'), row=1, col=2)

# indices = np.where(y == unique_classes[2])
# fig.add_trace(go.Scatter(x=tsne_results[indices, 0].flatten(), y=tsne_results[indices, 1].flatten(), mode='markers', name=f'Class {unique_classes[2]}'), row=2, col=1)

# indices = np.where(y == unique_classes[3])
# fig.add_trace(go.Scatter(x=tsne_results[indices, 0].flatten(), y=tsne_results[indices, 1].flatten(), mode='markers', name=f'Class {unique_classes[3]}'), row=2, col=2)

# fig.update_layout(height=1000, width=1000, title_text="t-SNE visualization by Class")
# fig.show()