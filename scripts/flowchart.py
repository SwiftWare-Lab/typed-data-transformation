# Create a new figure and axis with arrows between separation and the branches
fig, ax = plt.subplots(figsize=(8, 8))

# Turn off axes
ax.axis('off')

# Draw the rectangles for each step in the flowchart
ax.text(0.5, 0.85, 'Input Dataset', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
ax.text(0.5, 0.7, 'Identify Consecutive Similar Values', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgreen'))
ax.text(0.5, 0.55, 'Separate Consecutive and Non-consecutive Values', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightyellow'))

# Branch 1: Consecutive Values
ax.text(0.2, 0.4, 'Consecutive Values', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightcoral'))
ax.text(0.2, 0.25, 'Compress with RLE', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

# Branch 2: Non-consecutive Values
ax.text(0.8, 0.4, 'Non-consecutive Values', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightcoral'))
ax.text(0.8, 0.25, 'Decompose and Compress', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

# Final Step
ax.text(0.5, 0.1, 'Measure Compression Ratio', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

# Draw arrows
ax.annotate('', xy=(0.5, 0.77), xytext=(0.5, 0.82), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.5, 0.62), xytext=(0.5, 0.67), arrowprops=dict(facecolor='black', shrink=0.05))

# Arrows from separation to Consecutive and Non-consecutive Values
ax.annotate('', xy=(0.35, 0.47), xytext=(0.45, 0.52), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.65, 0.47), xytext=(0.55, 0.52), arrowprops=dict(facecolor='black', shrink=0.05))

# Draw arrows between branches and final step
ax.annotate('', xy=(0.2, 0.32), xytext=(0.2, 0.37), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.8, 0.32), xytext=(0.8, 0.37), arrowprops=dict(facecolor='black', shrink=0.05))

# Arrows to final step
ax.annotate('', xy=(0.5, 0.18), xytext=(0.35, 0.23), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.5, 0.18), xytext=(0.65, 0.23), arrowprops=dict(facecolor='black', shrink=0.05))

# Save the updated flowchart with arrows
flowchart_with_arrows_image_path = "/mnt/data/flowchart_with_arrows.png"
plt.savefig(flowchart_with_arrows_image_path, bbox_inches='tight')

# Display the image path for download
flowchart_with_arrows_image_path
