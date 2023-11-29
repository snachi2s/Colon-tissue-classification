import cv2
import matplotlib.pyplot as plt

def compute_color_histogram(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute histograms for each color channel
    hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    return hist_red, hist_green, hist_blue

def plot_histogram(hist, color, channel):
    plt.plot(hist, color=color)
    plt.title(f'{channel} Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # Provide the path to your image
    image_path = r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg'

    # Compute color histograms
    hist_red, hist_green, hist_blue = compute_color_histogram(image_path)

    # Plot histograms for each channel
    plot_histogram(hist_red, 'red', 'Red')
    plot_histogram(hist_green, 'green', 'Green')
    plot_histogram(hist_blue, 'blue', 'Blue')
