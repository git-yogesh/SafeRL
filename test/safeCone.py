import numpy as np
import matplotlib.pyplot as plt


def score_landing_path(points, cone_angle=5):
    """
    Calculate the score of a landing based on how many points lie within a defined cone, where the cone is defined by the vertical distance.

    :param points: List of tuples (x, y) representing the landing path.
    :param cone_angle: Angle of the cone in degrees.
    :return: Score (percentage of points within the cone).
    """
    # Convert the angle to radians and compute the tangent
    tan_theta = np.tan(np.radians(cone_angle / 2))

    # The landing point (tip of the cone) is at (0, 0)
    x0, y0 = 0, 0

    # Calculate distances from the landing point and check if within cone
    inside_cone = 0
    total_points = len(points)
    for x, y in points:
        # Calculate the vertical distance from the cone's tip
        vertical_distance = abs(y0 - y)
        # Maximum allowable horizontal distance for the current vertical position
        max_horizontal_distance = vertical_distance * tan_theta
        # Check if the point is within the cone based on its horizontal position
        print(x, y, vertical_distance, max_horizontal_distance)
        if abs(x - x0) <= max_horizontal_distance:
            inside_cone += 1

    # Calculate the score
    score = (inside_cone / total_points) * 100

    # Optionally, plot the points and cone for visualization
    plot_landing_path(points, cone_angle, x0, y0, tan_theta, score)

    return score


def plot_landing_path(points, cone_angle, x0, y0, tan_theta, score):
    """
    Plot the landing path and the cone for visualization, including the score on the plot.
    """
    plt.figure(figsize=(8, 6))
    x_vals, y_vals = zip(*points)
    plt.scatter(x_vals, y_vals, color='blue', label='Path points')

    # Generate cone lines
    y_range = np.linspace(0, 100, 300)
    x_upper = (y_range - y0) * tan_theta
    x_lower = -(y_range - y0) * tan_theta

    plt.plot(x_upper, y_range, 'r--', label=f'Cone boundary ({cone_angle}°)')
    plt.plot(x_lower, y_range, 'r--')
    plt.scatter([x0], [y0], color='green', s=100, label='Landing point')
    plt.xlabel('Horizontal position')
    plt.ylabel('Vertical position')
    plt.title('Landing Path and Cone')
    plt.legend()
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(100, 0)
    plt.gca().invert_yaxis()  # Invert y-axis to simulate descent
    plt.text(x0, 20, f'Score: {score:.2f}%', fontsize=12, verticalalignment='bottom', horizontalalignment='center')
    plt.show()


# Example usage
points = [(0, 100), (-1, 80), (-2, 60), (-3, 45), (-4, 30), (-3, 20), (-2, 10), (-1, 5), (0, 0)]
score = score_landing_path(points)
print(f"Score: {score:.2f}%")
