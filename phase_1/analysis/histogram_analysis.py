import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def process_folder(folder_path, bins):
    hist_red, hist_green, hist_blue = np.zeros(bins), np.zeros(bins), np.zeros(bins)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg')):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i, color_hist in enumerate([hist_red, hist_green, hist_blue]):
                color_hist += cv2.calcHist([image], [i], None, [bins], [0, 256]).flatten() #for all images channelwise 

    hist_red /= hist_red.sum()
    hist_green /= hist_green.sum()
    hist_blue /= hist_blue.sum()

    return hist_red, hist_green, hist_blue

def plot_histograms_plotly(hist_red, hist_green, hist_blue, folder_name, bins):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Red Channel", "Green Channel", "Blue Channel"))

    fig.add_trace(
        go.Bar(x=list(range(bins)), y=hist_red, marker_color='red'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=list(range(bins)), y=hist_green, marker_color='green'),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=list(range(bins)), y=hist_blue, marker_color='blue'),
        row=1, col=3
    )

    fig.update_layout(
        title=f'{folder_name} color histogram',
        xaxis_title="Bins",
        yaxis_title="Frequency",
    )
    fig.show()

# def plot_histograms(hist_red, hist_green, hist_blue, folder_name, bins):
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.bar(range(bins), hist_red, color='red')
#     plt.title('Red Channel')

#     plt.subplot(1, 3, 2)
#     plt.bar(range(bins), hist_green, color='green')
#     plt.title('Green Channel')

#     plt.subplot(1, 3, 3)
#     plt.bar(range(bins), hist_blue, color='blue')
#     plt.title('Blue Channel')

#     plt.savefig(f'{folder_name}_bins128.png')  
#     plt.show()

if __name__ == '__main__':    
    for folder_name in ['Normal_Tissue', 'Serrated_Lesion', 'Adenocarcinoma', 'Adenoma']:
        folder_path = os.path.join('./data', folder_name)
        hist_red, hist_green, hist_blue = process_folder(folder_path, bins=128)
        #plot_histograms(hist_red, hist_green, hist_blue, folder_name, bins=128)
        plot_histograms_plotly(hist_red, hist_green, hist_blue, folder_name, bins=128)

#pip install plotly==5.18.0