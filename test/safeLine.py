import numpy as np
import matplotlib.pyplot as plt

def fit_line(points):
    """Fit a line to given points using linear regression and return slope and intercept."""
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    return m, c

def distance_from_line(point, slope, intercept):
    """Calculate the perpendicular distance of a point from a line given by slope and intercept."""
    x, y = point
    return abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

def generate_points(n, noise=1.0):
    """Generate n points around a line y = 2x + 1 with added noise."""
    x = np.linspace(0, 100, n)
    y = 2 * x + 1 + np.random.normal(size=n, scale=noise)
    return list(zip(x, y))

def plot_points_line_threshold(points, slope, intercept, threshold):
    """Plot points, the best-fit line, and threshold lines."""
    x_coords, y_coords = zip(*points)
    distances = [distance_from_line(point, slope, intercept) for point in points]
    violations = [max(0, dist - threshold) for dist in distances]
    score = -sum(violations)  # Negative score for visualization

    # Plot points
    plt.scatter(x_coords, y_coords, color='blue', label='Points')

    # Plot best-fit line
    x_line = np.array([min(x_coords), max(x_coords)])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r', label='Best-fit line')

    # Plot threshold lines
    plt.plot(x_line, y_line + threshold, 'g--', label=f'Threshold (+{threshold})')
    plt.plot(x_line, y_line - threshold, 'g--', label=f'Threshold (-{threshold})')

    # Labeling the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Points and Best-Fit Line with Threshold\nScore: {score:.2f}')
    plt.legend()
    plt.show()

# Generate 100 points
points = generate_points(100, noise=10)

# Fit line to points
slope, intercept = fit_line(points)

# Threshold distance from the line
threshold = 10

# Plot everything
plot_points_line_threshold(points, slope, intercept, threshold)
