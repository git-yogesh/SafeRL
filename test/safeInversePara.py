import numpy as np
import matplotlib.pyplot as plt


def parabola(x, k=0.01, h=0, v=0):
    """Calculate parabolic curve values."""
    return k * (x - h) ** 2 + v


def check_within_parabolas(x, y, k=0.01):
    """Check if a point is within the bounds of the mirrored parabolas."""
    parabola_y = parabola(x, k)
    return y <= parabola_y


def score_landing_path(points, k=0.01):
    """
    Score the landing path based on avoidance of entering the parabolic no-entry zones.

    :param points: List of tuples (x, y) representing the landing path.
    :param k: Coefficient controlling the parabolas' curvature.
    :return: Score as a percentage of points that do not enter the parabolas.
    """
    outside_parabolas = 0
    total_points = len(points)

    for x, y in points:
        if not check_within_parabolas(x, y, k) and not check_within_parabolas(-x, y, k):
            outside_parabolas += 1

    # Calculate the score
    score = (outside_parabolas / total_points) * 100

    # Optionally, plot the path and parabolas
    plot_landing_path(points, k, score)

    return score


def plot_landing_path(points, k, score):
    """Visualize the path, parabolic boundaries, and display the score."""
    plt.figure(figsize=(10, 8))
    x_vals, y_vals = zip(*points)
    plt.scatter(x_vals, y_vals, color='blue', label='Path points')

    x_range = np.linspace(-10, 10, 400)
    plt.plot(x_range, parabola(x_range, k), 'r--', label='Parabolic boundary')
    plt.plot(x_range, parabola(-x_range, k), 'r--')  # Mirrored parabola

    plt.xlabel('Horizontal Position')
    plt.ylabel('Vertical Position')
    plt.title(f'Landing Path Evaluation\nScore: {score:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(0, 100)
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization of descent
    plt.show()


# Define the path of the lander as an example
points = [(-9, 90), (-8, 82), (-6, 70), (-4, 60), (-2, 52), (0, 45), (2, 40), (4, 35), (6, 30), (8, 26), (9, 23)]

# Calculate and print the score
score = score_landing_path(points)
print(f"Score: {score:.2f}%")
