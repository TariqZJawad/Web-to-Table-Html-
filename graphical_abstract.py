import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import os
import matplotlib.image as mpimg
from PIL import Image

# ==========================================
# Configuration and Styling (Light Mode)
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'  # For math symbols that look like Times

# Define colors for light theme
C_BLACK = '#000000'
C_DARK_GREY = '#333333'
C_LIGHT_GREY = '#F0F0F0'
C_CAVITY = '#A9CCE3'  # Lighter blue for cavity box
C_QUBIT = '#F1948A'   # Lighter red for qubits
C_HIGHLIGHT = '#F7DC6F' # Soft yellow
C_TEXT = C_BLACK
C_BG = '#FFFFFF'     # White background

# Define required image files
REQUIRED_IMAGES = ['f1.png', 'f2.png', 'f3.png', 'f.png']

# Placeholder function for missing images
def create_placeholder_image(filename, text, size=(800, 600)):
    """Creates a placeholder PNG image if the file is missing."""
    img = Image.new('RGB', size, color=C_LIGHT_GREY)
    # Using basic PIL drawing to create placeholder content (text)
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30) # Try standard font
    except IOError:
        font = ImageFont.load_default() # Fallback
    
    # تم تصحيح الخطأ هنا باستخدام size[1] بدلاً من height_p
    d.text((50, size[1]//2 - 30), f"Missing File: {filename}\n{text}", fill=C_BLACK, font=font)
    img.save(filename)
    print(f"Created placeholder image: {filename}")

# Check for required images and create placeholders if needed
for img_file in REQUIRED_IMAGES:
    if not os.path.exists(img_file):
        create_placeholder_image(img_file, f"Place Fig in file {img_file}")

# Load image loading function
def load_image(ax, filename, padding=0.0):
    """Loads an image into a specified matplotlib axes, preserving aspect ratio."""
    try:
        img = mpimg.imread(filename)
        ax.imshow(img, aspect='equal', interpolation='nearest') # 'equal' aspect to preserve ratio
        ax.axis('off') # Hide axes ticks/labels
    except FileNotFoundError:
        ax.text(0.5, 0.5, f'Missing:\n{filename}', ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')

# ==========================================
# Figure Initialization
# ==========================================
fig = plt.figure(figsize=(24, 13.5), facecolor=C_BG) # Landscape 16:9

# ==========================================
# 1. Header Section (Top)
# ==========================================
# Main Title
fig.text(0.5, 0.96, "Ergotropy Protection via Cavity Detuning in Collective Open Quantum Batteries", 
         fontsize=28, weight='bold', color=C_BLACK, ha='center', va='center')

# Author Info
fig.text(0.5, 0.92, "(Tariq Zeyad Jawad, University of Kufa, Kufa, Najaf, Iraq)", 
         fontsize=20, weight='normal', color=C_DARK_GREY, ha='center', va='center')

# ==========================================
# 2. Problem Section (Far Left, 2 rows)
# ==========================================
gs_problem = fig.add_gridspec(2, 1, left=0.02, right=0.18, bottom=0.05, top=0.88, hspace=0.1)

# --- Row 1: Schematic 1 (Rapid Loss) ---
ax_p1 = fig.add_subplot(gs_problem[0, 0], facecolor=C_BG)
ax_p1.set_title("Problem", fontsize=22, color=C_BLACK, loc='left', pad=15, fontweight='bold')
ax_p1.set_xlim(0, 1)
ax_p1.set_ylim(0, 1)
ax_p1.axis('off')

# Drawing basic Cavity box
cavity_rect_1 = patches.Rectangle((0.15, 0.1), 0.7, 0.8, linewidth=2, edgecolor=C_CAVITY, facecolor='none')
ax_p1.add_patch(cavity_rect_1)
ax_p1.text(0.5, 0.03, r"Cavity Box $(\omega_c)$", ha='center', va='center', fontsize=16, color=C_TEXT)

# Drawing 4 circles (qubits) inside a smaller box named "qubits"
qubits_rect_1 = patches.Rectangle((0.3, 0.25), 0.4, 0.5, linewidth=1, edgecolor=C_QUBIT, facecolor='none')
ax_p1.add_patch(qubits_rect_1)
ax_p1.text(0.5, 0.77, r"$\omega_b = \omega_c$", ha='center', va='center', fontsize=18, color=C_TEXT)

for i in range(4):
    y_pos = 0.35 + i*0.13
    circle = patches.Circle((0.5, y_pos), 0.04, linewidth=1.5, edgecolor=C_TEXT, facecolor=C_QUBIT)
    ax_p1.add_patch(circle)
    ax_p1.text(0.5, y_pos, r'$\sigma$', ha='center', va='center', fontsize=16, color=C_TEXT)

# Text: loss ergotropy rapidlly
ax_p1.text(0.5, -0.07, "loss ergotropy rapidly", ha='center', va='center', fontsize=16, color='red', fontweight='bold')

# Basic arrows indicating loss / interaction
ax_p1.annotate('', xy=(0.8, 0.5), xytext=(0.55, 0.5), arrowprops=dict(facecolor=C_TEXT, shrink=0.05, width=1, headwidth=5))
ax_p1.annotate('', xy=(0.2, 0.5), xytext=(0.45, 0.5), arrowprops=dict(facecolor=C_TEXT, shrink=0.05, width=1, headwidth=5))

# --- Row 2: Schematic 2 (Bended Arrows) ---
ax_p2 = fig.add_subplot(gs_problem[1, 0], facecolor=C_BG)
ax_p2.set_xlim(0, 1)
ax_p2.set_ylim(0, 1)
ax_p2.axis('off')

# Drawing primary Cavity box again
cavity_rect_2 = patches.Rectangle((0.15, 0.1), 0.7, 0.8, linewidth=2, edgecolor=C_CAVITY, facecolor='none')
ax_p2.add_patch(cavity_rect_2)

# Drawing smaller box inside named w_b
wb_rect = patches.Rectangle((0.35, 0.3), 0.3, 0.4, linewidth=1.5, edgecolor=C_HIGHLIGHT, facecolor='none')
ax_p2.add_patch(wb_rect)
ax_p2.text(0.5, 0.35, r"$w_b$ box", ha='center', va='center', fontsize=16, color=C_TEXT)

# Drawing 4 circles (qubits) again inside a smaller inner container for visual grouping
qubits_inner_rect_2 = patches.Rectangle((0.42, 0.38), 0.16, 0.24, linewidth=1, edgecolor=C_QUBIT, facecolor='none')
ax_p2.add_patch(qubits_inner_rect_2)
for i in range(4):
    y_pos = 0.4 + i*0.07
    circle = patches.Circle((0.5, y_pos), 0.025, linewidth=1.2, edgecolor=C_TEXT, facecolor=C_QUBIT)
    ax_p2.add_patch(circle)
    ax_p2.text(0.5, y_pos, r'$\sigma$', ha='center', va='center', fontsize=12, color=C_TEXT)

# Bended arrows simulation: complex paths kinked at boundaries
# Arrows entering outer, breaking into inner
# Path: Outer Entry -> Outer Wall -> Inner Wall -> Inner Target (breaking path)
# Simplification using kinked arrows (using annotate with connectionstyle)
ax_p2.annotate('', xy=(0.42, 0.5), xytext=(0.05, 0.5), # Arrow 1: external straight to outer wall entry boundary (conceptual)
             arrowprops=dict(arrowstyle="->", facecolor=C_TEXT, edgecolor=C_TEXT, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
# Arrow that "breaks" visually at inner boundary
ax_p2.annotate('', xy=(0.6, 0.6), xytext=(0.42, 0.5), # From outer wall boundary kink to inside inner box
             arrowprops=dict(arrowstyle="->", facecolor=C_TEXT, edgecolor=C_TEXT, connectionstyle="arc3,rad=.2"))

# Arrows leaving inner, breaking to outer
ax_p2.annotate('', xy=(0.85, 0.5), xytext=(0.58, 0.5), # External straight path conceptually leaving outer wall boundary
             arrowprops=dict(arrowstyle="<-", facecolor=C_TEXT, edgecolor=C_TEXT, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
# Arrow that "breaks" visually at outer boundary when leaving inner
ax_p2.annotate('', xy=(0.7, 0.6), xytext=(0.85, 0.5), # Kink when leaving outer wall conceptual boundary
             arrowprops=dict(arrowstyle="->", facecolor=C_TEXT, edgecolor=C_TEXT, connectionstyle="arc3,rad=.2"))


# Text: دالة delta below everything
ax_p2.text(0.5, -0.07, r"Delta Function ($\delta$)", ha='center', va='center', fontsize=16, color=C_TEXT)

# ==========================================
# 3. Figures Section (Main Content, 3 Columns)
# ==========================================
gs_figures = fig.add_gridspec(2, 3, left=0.20, right=0.78, bottom=0.05, top=0.88, wspace=0.1, hspace=0.1)

# Column 1: f1.png
ax_f1 = fig.add_subplot(gs_figures[0:2, 0], facecolor=C_BG) # Spans 2 rows
load_image(ax_f1, 'f1.png')

# Column 2: f2.png
ax_f2 = fig.add_subplot(gs_figures[0:2, 1], facecolor=C_BG) # Spans 2 rows
load_image(ax_f2, 'f2.png')

# Column 3: f3.png (top half), f.png (bottom half)
ax_f3 = fig.add_subplot(gs_figures[0, 2], facecolor=C_BG)
load_image(ax_f3, 'f3.png')

ax_f = fig.add_subplot(gs_figures[1, 2], facecolor=C_BG)
load_image(ax_f, 'f.png')

# ==========================================
# 4. Key Results Section (Right Panel)
# ==========================================
gs_results = fig.add_gridspec(1, 1, left=0.80, right=0.98, bottom=0.05, top=0.88)
ax_results = fig.add_subplot(gs_results[0, 0], facecolor=C_LIGHT_GREY) # Use light grey for results panel distinction
ax_results.set_title("Key Results", fontsize=22, color=C_BLACK, weight='bold', pad=20)
ax_results.set_xlim(0, 1)
ax_results.set_ylim(0, 1)
ax_results.axis('off')

# List of synthesized Key Results based on earlier analysis
results_list = [
    r"1. Optimal detuning $\Delta^*(N) \propto N^{1/2}$ established analytically and numerically verified.",
    r"2. Resolve Non-Markovian Paradox: Passive detuning suppresses memory but improves ergotropy (up to 1088%).",
    r"3. Collective quantum advantage threshold identified at $N \geq 3$ for realistic noise models.",
    r"4. Quantitative breakdown ceiling of Tavis-Cummings RWA approximation provided ($N_{max} \approx 25$).",
    r"5. Proposed passive control mechanism demonstrates efficient energy protection with no external power cost.",
]

# Write results text to the panel
for i, result_text in enumerate(results_list):
    y_pos_text = 0.85 - i*0.12 # Spacing between lines
    ax_results.text(0.05, y_pos_text, result_text, 
                     fontsize=16, color=C_TEXT, ha='left', va='center', 
                     wrap=True) # wrap=True helpful for long lines

# ==========================================
# Final Figure Adjustment and Save
# ==========================================
out_path = 'graphical_abstract.png'
plt.tight_layout()
fig.savefig(out_path, dpi=300, facecolor=C_BG, edgecolor='none', bbox_inches='tight') #bbox_inches='tight' removes extra padding

print(f"Graphical abstract successfully generated and saved to: {os.path.abspath(out_path)}")
