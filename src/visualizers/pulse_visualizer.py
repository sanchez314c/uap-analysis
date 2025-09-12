import matplotlib.pyplot as plt
import numpy as np

# Create time array and data arrays
t = np.linspace(0, 39.79, 955)
base_intensity = np.zeros_like(t)
pulse_freq = 2.3  # Hz
mod_freq = 0.5   # Hz

# Generate light intensity data
for i, time in enumerate(t):
    base = np.sin(2 * np.pi * pulse_freq * time)
    mod = 0.3 * np.sin(2 * np.pi * mod_freq * time)
    base_intensity[i] = 0.4 + 0.6 * (0.7 + 0.3 * base + mod)

# Create figure
plt.figure(figsize=(15, 6))
plt.plot(t, base_intensity, "g-", linewidth=1, label="Light Intensity")
plt.grid(True, alpha=0.3)
plt.title("UAP Light Pulse Pattern Analysis")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Light Intensity")
plt.ylim(0, 1)

# Add key events markers
events = [
    (5.29, "UAP Appears"),
    (7.20, "Acceleration"),
    (12.16, "Direction Change"),
    (23.25, "Secondary Acceleration"),
    (34.12, "UAP Exits")
]

for time, label in events:
    plt.axvline(x=time, color="r", linestyle="--", alpha=0.3)
    plt.text(time, 0.95, label, rotation=90, verticalalignment="top")

plt.tight_layout()
plt.savefig("/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/pulse_pattern_final.png", dpi=300, bbox_inches="tight")
