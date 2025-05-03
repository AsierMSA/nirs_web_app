import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

# Main platform box
main_box = FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle="round,pad=0.03", 
                          linewidth=2, edgecolor='#2a4d69', facecolor='#f4f4f4', alpha=0.95)
ax.add_patch(main_box)
ax.text(0.5, 0.75, "NIRS Data Analysis Platform", fontsize=18, fontweight='bold', ha='center', va='center', color='#2a4d69')

# Input data
ax.add_patch(Rectangle((0.13, 0.62), 0.18, 0.12, linewidth=1.5, edgecolor='#4b86b4', facecolor='#dbe9f6'))
ax.text(0.22, 0.68, ".fif.gz NIRS Data", fontsize=12, ha='center', va='center', color='#4b86b4')

# Signal processing
ax.add_patch(Rectangle((0.13, 0.45), 0.18, 0.12, linewidth=1.5, edgecolor='#63ace5', facecolor='#eaf6fb'))
ax.text(0.22, 0.51, "Signal Processing\n(Hemodynamic Analysis)", fontsize=11, ha='center', va='center', color='#63ace5')

# ML classification
ax.add_patch(Rectangle((0.41, 0.45), 0.18, 0.12, linewidth=1.5, edgecolor='#b4c6e7', facecolor='#f7fbfc'))
ax.text(0.50, 0.51, "Machine Learning\n(Classification)", fontsize=11, ha='center', va='center', color='#2a4d69')

# Temporal bias validation
ax.add_patch(Rectangle((0.69, 0.45), 0.18, 0.12, linewidth=1.5, edgecolor='#f18d9e', facecolor='#fbeff2'))
ax.text(0.78, 0.51, "Temporal Bias\nValidation", fontsize=11, ha='center', va='center', color='#f18d9e')

# Visualization & interpretation
ax.add_patch(Rectangle((0.41, 0.28), 0.18, 0.12, linewidth=1.5, edgecolor='#7bc043', facecolor='#eafaf1'))
ax.text(0.50, 0.34, "Visualization &\nInterpretation", fontsize=11, ha='center', va='center', color='#7bc043')

# Arrows
arrowprops = dict(arrowstyle="->", color='#2a4d69', lw=2)
ax.annotate("", xy=(0.22, 0.62), xytext=(0.22, 0.57), arrowprops=arrowprops)
ax.annotate("", xy=(0.32, 0.51), xytext=(0.41, 0.51), arrowprops=arrowprops)
ax.annotate("", xy=(0.59, 0.51), xytext=(0.69, 0.51), arrowprops=arrowprops)
ax.annotate("", xy=(0.50, 0.45), xytext=(0.50, 0.40), arrowprops=arrowprops)

# Limitations box
ax.add_patch(FancyBboxPatch((0.1, 0.05), 0.8, 0.10, boxstyle="round,pad=0.02", 
                            linewidth=1.5, edgecolor='#f37736', facecolor='#fff3e6', alpha=0.9))
ax.text(0.5, 0.10, "Limitations: No real-time acquisition, no EEG/fMRI integration. Focused on offline NIRS analysis.", 
        fontsize=11, ha='center', va='center', color='#f37736')

plt.tight_layout()
plt.show()