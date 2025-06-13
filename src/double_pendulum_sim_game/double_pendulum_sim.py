# %%
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy import cos, sin

# Physical constants
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 0.69  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 0.69  # mass of pendulum 2 in kg
DAMPING_COEFFICIENT = 0.1  # damping coefficient for angular velocity


def derivs(
    state: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Calculate derivatives for double pendulum differential equation with damping."""
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) * cos(delta)

    # Add damping force proportional to angular velocity
    damping_force_1 = -DAMPING_COEFFICIENT * state[1]

    dydx[1] = (
        M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
        + M2 * G * sin(state[2]) * cos(delta)
        + M2 * L2 * state[3] * state[3] * sin(delta)
        - (M1 + M2) * G * sin(state[0])
        + damping_force_1 * den1  # Apply damping
    ) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1

    # Add damping force for second pendulum
    damping_force_2 = -DAMPING_COEFFICIENT * state[3]

    dydx[3] = (
        -M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
        + (M1 + M2) * G * sin(state[0]) * cos(delta)
        - (M1 + M2) * L1 * state[1] * state[1] * sin(delta)
        - (M1 + M2) * G * sin(state[2])
        + damping_force_2 * den2  # Apply damping
    ) / den2

    return dydx


def simulate_pendulum(
    initial_angles: tuple[float, float],
    initial_velocities: tuple[float, float],
    t_stop: float = 50,
    dt: float = 0.01,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.floating]],
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
]:
    """Simulate double pendulum motion using Euler's method."""
    t = np.arange(0, t_stop, dt)
    th1, th2 = initial_angles
    w1, w2 = initial_velocities

    # Convert to radians and create initial state
    state: np.ndarray[tuple[int], np.dtype[np.floating]] = np.radians([th1, w1, th2, w2])

    # Integrate using Euler's method
    y: np.ndarray[tuple[int, int], np.dtype[np.floating]] = np.empty((len(t), 4))
    y[0] = state
    for i in range(1, len(t)):
        y[i] = y[i - 1] + derivs(y[i - 1]) * dt

    return t, y


def calculate_positions(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Cartesian positions from angular positions."""
    x1 = L1 * sin(y[:, 0])
    y1 = -L1 * cos(y[:, 0])
    x2 = L2 * sin(y[:, 2]) + x1
    y2 = -L2 * cos(y[:, 2]) + y1
    return x1, y1, x2, y2


def setup_animation_figure() -> tuple[Figure, Axes, Line2D, Line2D, Text]:
    """Set up matplotlib figure and animation elements."""
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.0))
    ax.set_aspect("equal")
    ax.grid()

    (line,) = ax.plot([], [], "o-", lw=2)
    (trace,) = ax.plot([], [], ".-", lw=1, ms=2)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    return fig, ax, line, trace, time_text


def create_animation_function(  # noqa: PLR0913
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    line: Line2D,
    trace: Line2D,
    time_text: Text,
    dt: float,
) -> Callable[[int], tuple[Line2D, Line2D, Text]]:
    """Create animation function for matplotlib FuncAnimation."""

    def animate(i: int) -> tuple[Line2D, Line2D, Text]:
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(f"time = {i * dt:.1f}s")
        return line, trace, time_text

    return animate


def run_double_pendulum_simulation() -> None:
    """Run the complete double pendulum simulation and animation."""
    # Initial conditions
    initial_angles = (120.0, -10.0)  # th1, th2 in degrees
    initial_velocities = (0.0, 0.0)  # w1, w2 in degrees per second
    t_stop = 50
    dt = 0.01

    # Run simulation
    t, y = simulate_pendulum(initial_angles, initial_velocities, t_stop, dt)

    # Calculate positions
    x1, y1, x2, y2 = calculate_positions(y)

    # Setup animation
    fig, ax, line, trace, time_text = setup_animation_figure()
    animate_func = create_animation_function(x1, y1, x2, y2, line, trace, time_text, dt)

    # Create and run animation
    ani = animation.FuncAnimation(  # noqa: F841
        fig=fig,
        func=animate_func,
        frames=len(y),
        interval=dt * 1000,
        blit=True,
    )
    plt.show()


# Run the simulation
run_double_pendulum_simulation()
