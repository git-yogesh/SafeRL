import numpy as np
import matplotlib.pyplot as plt


def score_landing_path(points, x0, y0, a=0.2):
    """
    Calculate the score of a landing based on how many points lie within a defined parabolic cone.

    :param points: List of tuples (x, y) representing the landing path.
    :param a: Coefficient that controls the spread of the parabola.
    :return: Score (percentage of points within the parabola).
    """
    # The landing point (tip of the parabola) is at (0, 0)

    # Calculate distances from the landing point and check if within parabola
    inside_parabola = 0
    total_points = len(points)
    for x, y in points:
        # Calculate the vertical distance from the parabola's tip
        vertical_distance = y0 - y
        # Maximum allowable horizontal distance for the current vertical position
        max_horizontal_distance = np.sqrt(a * vertical_distance)
        # Check if the point is within the parabola based on its horizontal position
        if abs(x - x0) <= max_horizontal_distance:
            inside_parabola += 1

    # Calculate the score
    score = (inside_parabola / total_points) * 100

    # Optionally, plot the points and parabola for visualization
    plot_landing_path(points, a, x0, y0, score)

    return score


def plot_landing_path(points, a, x0, y0, score):
    """
    Plot the landing path and the parabolic cone for visualization, including the score on the plot.
    """
    plt.figure(figsize=(8, 6))
    x_vals, y_vals = zip(*points)
    plt.scatter(x_vals, y_vals, color='blue', label='Path points')

    y_range = np.linspace(y0, y0 + 100, 300)
    x_upper = x0 + np.sqrt(a * (y_range - y0))
    x_lower = x0 - np.sqrt(a * (y_range - y0))
    plt.plot(x_upper, y_range, 'r--', label=f'Parabola boundary (a={a})')
    plt.plot(x_lower, y_range, 'r--')


    plt.scatter([x0], [y0], color='green', s=100, label='Landing point')
    plt.scatter([-0.2, 0.2], [0, 0], color='red', s=100, label='Boundaries')

    plt.xlabel('Horizontal position')
    plt.ylabel('Vertical position')
    plt.title('Landing Path and Parabolic Cone')
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(max(y_vals), min(y_vals))
    plt.gca().invert_yaxis()  # Invert y-axis to simulate descent
    plt.text(x0, 1.5, f'Score: {score:.2f}%', fontsize=12, verticalalignment='bottom', horizontalalignment='center')
    plt.show()


# Example usage
# points = [(0, 100), (-1, 80), (-2, 60), (-3, 45), (-4, 30), (-3, 20), (-2, 10), (-1, 5), (0, 0)]
# score = score_landing_path(points, a=0.167)
# print(f"Score: {score:.2f}%")
