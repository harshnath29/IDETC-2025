import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


def CannyProcess(image_path, color_range) -> list:
    src = cv.imread(image_path)
    img_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    
    lower_bound = np.array(color_range[0])
    upper_bound = np.array(color_range[1])
    
    mask = cv.inRange(img_hsv, lower_bound, upper_bound)
    
    # Apply Gaussian blur instead of regular blur
    blurred = cv.GaussianBlur(mask, (5, 5), 0)
    
    # Use Canny edge detection with lower thresholds
    edges = cv.Canny(blurred, 50, 150)
    
    # Perform morphological operations to refine edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    return edges

def createContours(edges: list, drawContours = False) -> list:
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    
    if drawContours:
        drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for contour in contours:
            # Use sub-pixel accuracy for drawing contours
            refined_contour = cv.approxPolyDP(contour, 0.1, True)
            cv.drawContours(drawing, [refined_contour], 0, (255, 255, 255), 1)
        
        cv.namedWindow("Contours", cv.WINDOW_NORMAL)
        cv.imshow('Contours', drawing)
        cv.waitKey()
    
    return contours

def compare_contours(contours1, contours2, distance_threshold=5, y_step=5):
    """
    Compares two contours along the Y-axis, measuring the distance between the farthest X on the left print
    and the nearest X on the right print for each Y-axis level.

    :param contours1: List of contours from the first (left) print
    :param contours2: List of contours from the second (right) print
    :param distance_threshold: Max allowed distance for a continuous boundary
    :param y_step: Step size for sampling along the Y-axis
    :return: Boolean (True if boundary is continuous, False if gaps exist)
    """
    # Convert contours to x, y coordinates
    x1, y1, x2, y2 = [], [], [], []

    for contour in contours1:
        for point in contour:
            row, col = point[0]
            x1.append(row)
            y1.append(-1 * col)

    for contour in contours2:
        for point in contour:
            row, col = point[0]
            x2.append(row)
            y2.append(-1 * col)

    # Convert to numpy arrays for easier manipulation
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)

    # Y-axis range for sampling
    min_y = max(min(y1), min(y2))
    max_y = min(max(y1), max(y2))
    y_samples = np.arange(min_y, max_y, y_step)

    result_array = []
    for y in y_samples:
        # Get points near current Y level for both contours
        points1_y = x1[np.abs(y1 - y) <= y_step]
        points2_y = x2[np.abs(y2 - y) <= y_step]

        if len(points1_y) == 0 or len(points2_y) == 0:
            continue  # Skip if no points found at this Y-level

        # Farthest X on left print (max X for left)
        max_x1 = np.max(points1_y)

        # Nearest X on right print (min X for right)
        min_x2 = np.min(points2_y)

        # Calculate distance
        distance = min_x2 - max_x1

        # Append result based on distance threshold
        if distance < 30:
            result_array.append(distance <= distance_threshold)

    # Evaluate the majority of the results
    if len(result_array) == 0:
        print("Insufficient data for comparison.")
        return False

    pass_ratio = np.sum(result_array) / len(result_array)
    print(f"Pass Ratio: {pass_ratio:.2f}")

    return pass_ratio > 0.5  # Adjust pass ratio as needed

def maximumHeight(contours: list) -> float:
    '''
    Returns the maximum height/y-value of the input contours list
    :param contours: contours for the image to analyze
    :return: Maximum height of all points detected (float)
    '''
    # First: convert to x, y coordinates
    x1, y1 = list(), list()
    for i in contours:
        for j in i:
            row, col = j[0]
            x1.append(row)
            y1.append(-1 * col)

    return max(y1) * -1


def compareBoundingEdges(contours1: list, contours2: list, cutHeight: float, showCutImage = False) -> dict:
    '''
    Compares the two images' bounding boxes
    :param contours1: List of contours for the first image
    :param contours2: List of contours for the second image
    :param cutHeight: Data above this height will be ignored
    :return: A dictionary containing the number of pixel difference between the corners of each bounding box for each image
    '''
    # First: convert to x, y coordinates
    x1, y1, x2, y2 = list(), list(), list(), list()
    for i in contours1:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                x1.append(row)
                y1.append(-1 * col)

    for i in contours2:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                x2.append(row)
                y2.append(-1 * col)

    if showCutImage:
        plt.scatter(x1, y1, s = 1, color = 'red')
        plt.scatter(x2, y2, s = 1, color = 'blue')
        plt.title('Cut Image'), plt.xticks([]), plt.yticks([])
        plt.xlim(0, 2500)
        plt.ylim(-3000, 0)
        plt.show()

    # Find bounding box coordinates
    x1min, x1max, y1min, y1max = min(x1), max(x1), min(y1), max(y1)
    x2min, x2max, y2min, y2max = min(x2), max(x2), min(y2), max(y2)

    print(x1min, x1max, y1min, y1max)
    print(x2min, x2max, y2min, y2max)

    # Find differences
    differences = {'Leftmost Point' : x1min - x2min, 'Bottommost Point': y1min - y2min,
                   'Rightmost Point' : x1max - x2max, 'Topmost Point' : y1max - y2max}

    return differences


def closestPointComparison(contours1: list, contours2: list, cutHeight: float, distance_threshold = 15) -> int:
    '''
    Compares every point to its closest point on the other image.
    :param contours1: List of contours for the first image
    :param contours2: List of contours for the second image
    :param cutHeight: Data above this height will be ignored
    :return: Boolean stating whether the image is similar/warping is detected. TRUE means it is similar. FALSE means it is different.
    '''
    # First: convert to x, y coordinates
    xy1, xy2 = list(), list()
    for i in contours1:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                xy1.append((row, -1*col))

    for i in contours2:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                xy2.append((row, -1 * col))

    xy1, xy2 = np.array(xy1), np.array(xy2)

    max_distance = 0

    # Find closest point in xy2 for each point in xy1
    for point in xy1:
        distances = np.linalg.norm(xy2 - point, axis=1)
        min_distance = np.min(distances)
        max_distance = max(max_distance, min_distance)

    return max_distance


def kNNComparison(contours1: list, contours2: list, cutHeight: float, showCutImage = False):
    '''
    Compares two images using a kNN regression as I learned in SDS322E: Elements of Data Science. I left in for usage sake, but
    it doesn't seem to work too well.
    :param contours1: List of contours for the first image
    :param contours2: List of contours for the second image
    :param cutHeight: Data above this height will be ignored
    :return:
    '''
    # First: convert to x, y coordinates
    x1, y1, x2, y2 = list(), list(), list(), list()
    for i in contours1:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                x1.append(row)
                y1.append(-1 * col)

    for i in contours2:
        for j in i:
            row, col = j[0]
            if col > cutHeight:
                x2.append(row)
                y2.append(-1 * col)

    x1, y1, x2, y2 = np.array(x1).reshape(-1,1), np.array(y1).reshape(-1,1), np.array(x2).reshape(-1,1), np.array(y2).reshape(-1,1)
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(x1, y1)
    y_pred = knn_regressor.predict(x2)
    mse = mean_squared_error(y2, y_pred)
    r2 = r2_score(y2, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    plt.scatter(x2, y2, color='blue', label='Actual')
    plt.scatter(x2, y_pred, color='red', label='Predicted')
    plt.title('KNN Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


def plotEdgesonImage(image, edges):
    '''
    Compares the detected edges to the source image for troubleshooting
    :param image: source image
    :param edges: edges from Canny
    :return:
    '''
    plt.subplot(121), plt.imshow(image) # Note: colors swapped b/c OpenCV stores colors in BGR, but matplotlib does in RGB
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges)
    plt.title('Edges Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # Example of how to run these functions

    # Canny Edges / Contours
    edges1 = CannyProcess('images\z_height_3.25.jpg')
    edges2 = CannyProcess('images\z_height_6.45.jpg')

    contours1 = createContours(edges1)
    contours2 = createContours(edges2)

    # Get maximum height for one of these contours (can use this to determine cutHeight for comparison)
    print(maximumHeight(contours1))

    # Difference using Bounding Box Method
    differences = compareBoundingEdges(contours1, contours2, cutHeight = maximumHeight(contours1) + 80, showCutImage = True)
    print(differences)

    # Difference comparing Point by Point
    print(closestPointComparison(contours1, contours2, cutHeight = maximumHeight(contours1) + 40, distance_threshold= 5))