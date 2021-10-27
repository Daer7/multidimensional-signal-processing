import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(precision=3, suppress=True)


def interp_nearest(x_source, y_source, x_target):
    dist = x_source[:, np.newaxis] - x_target[np.newaxis, :]
    abs_dist = np.abs(dist)
    addr = np.argmin(abs_dist, axis=0)
    y_target = y_source[addr]
    return y_target
    # return np.zeros_like(x_target)


def interp_linear(x_source, y_source, x_target):
    dist = x_source[:, np.newaxis] - x_target[np.newaxis, :]
    dist_lower = np.copy(dist)
    dist_lower[dist < 0] = np.nan

    dist_upper = np.copy(dist)
    dist_upper[dist > 0] = np.nan

    addr_upper = np.nanargmax(dist_upper, axis=0)
    addr_lower = np.nanargmin(dist_lower, axis=0)

    y_1 = y_source[addr_upper]
    y_0 = y_source[addr_lower]

    x_1 = x_source[addr_upper]
    x_0 = x_source[addr_lower]

    A = (y_1 - y_0) / (x_1 - x_0)
    B = (y_0 * x_1 - y_1 * x_0) / (x_1 - x_0)

    y_target = A * x_target + B
    y_target[np.isnan(y_target)] = y_0[np.isnan(y_target)]
    # print(y_target)
    return y_target
    # return np.zeros_like(x_target)


def interp_cubic(x_source, y_source, x_target):
    m = x_source.size
    n = x_source.size - 1

    a = np.copy(y_source)
    b = np.empty(n)
    d = np.empty(n)
    h = np.diff(x_source)

    beta = np.diff(a) / h
    alpha = np.empty(n)
    for i in range(1, n):
        alpha[i] = 3 * beta[i] - beta[i - 1]

    c = np.empty(m)
    l = np.empty(m)
    mu = np.empty(m)
    z = np.empty(m)

    l[0] = 1
    mu[0] = z[0] = 0

    for i in range(1, n):
        l[i] = 2 * (x_source[i + 1] - x_source[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1
    z[n] = c[n] = 0

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = beta[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    dist = x_source[:, np.newaxis] - x_target[np.newaxis, :]
    abs_dist = np.abs(dist)
    addr = np.argmin(abs_dist, axis=0)
    addr[addr == n] = n - 1

    y_target = (
        a[addr]
        + b[addr] * (x_target - x_source[addr])
        + c[addr] * (x_target - x_source[addr]) ** 2
        + d[addr] * (x_target - x_source[addr]) ** 3
    )

    return y_target


# Probers
original_prober = np.sort(np.random.uniform(size=8) * np.pi * 4)
target_prober = np.linspace(np.min(original_prober), np.max(original_prober), 32)

# Sampling
original_signal = np.sin(original_prober)

# Out-of-box interpolators
fn = interp1d(original_prober, original_signal, kind="nearest")
fl = interp1d(original_prober, original_signal, kind="linear")
fc = interp1d(original_prober, original_signal, kind="cubic")

# Interpolation
target_signal_fn = fn(target_prober)
target_signal_fl = fl(target_prober)
target_signal_fc = fc(target_prober)

args = (original_prober, original_signal, target_prober)
own_target_signal_fn = interp_nearest(*args)
own_target_signal_fl = interp_linear(*args)
own_target_signal_fc = interp_cubic(*args)

# Store them for plotting
target_signals = [target_signal_fn, target_signal_fl, target_signal_fc]
own_target_signals = [own_target_signal_fn, own_target_signal_fl, own_target_signal_fc]

# Plotting
fig, ax = plt.subplots(4, 1, figsize=(8, 8 * 1.618))

ax[0].scatter(
    original_prober, np.ones_like(original_prober) * -0.5, label="origin", c="black"
)
ax[0].scatter(target_prober, np.ones_like(target_prober) * 0.5, label="target", c="red")
ax[0].plot(original_prober, original_signal, c="black")
ax[0].set_ylim(-1.5, 1.5)
ax[0].legend(frameon=False, loc=9, ncol=2)
ax[0].set_yticks([])
ax[0].set_xticks(original_prober)
ax[0].set_xticklabels([])
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_visible(False)
ax[0].grid(ls=":")


for i, (target_signal, own_target_signal) in enumerate(
    zip(target_signals, own_target_signals)
):
    ax[1 + i].plot(original_prober, original_signal, c="black", ls=":")
    ax[1 + i].plot(target_prober, target_signal, "red", ls=":")
    ax[1 + i].plot(target_prober, own_target_signal, "red")
    ax[1 + i].grid(ls=":")
    ax[1 + i].set_xticks(target_prober)
    ax[1 + i].set_xticklabels([])
    ax[1 + i].spines["top"].set_visible(False)
    ax[1 + i].spines["right"].set_visible(False)

ax[1].set_title("neighbor")
ax[2].set_title("linear")
ax[3].set_title("cubic")

plt.savefig("foo.png")
