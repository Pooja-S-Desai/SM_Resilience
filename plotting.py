# Plotting

import os
import math
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

from adjustText import adjust_text

from helpers import TOPOLOGY_FOLDER, TOPOLOGY_LIST_FILE, PROPAGATION_SPEED, ensure_dir, CAPACITY_THRESHOLD as delta


# -------------------------------------------------------------
# Save poster-quality figures
# -------------------------------------------------------------
def save_poster_figures(fig, base_path):

    svg_path = base_path + ".svg"
    pdf_path = base_path + ".pdf"
    jpg_path = base_path + ".jpg"

    fig.savefig(svg_path, bbox_inches="tight")
    print("✅ SVG saved:", svg_path)

    fig.savefig(pdf_path, bbox_inches="tight")
    print("✅ PDF saved:", pdf_path)

    fig.savefig(jpg_path, dpi=600, bbox_inches="tight")
    print("✅ JPG saved:", jpg_path)

    return svg_path, pdf_path, jpg_path


# -------------------------------------------------------------
# Extract geographical positions
# -------------------------------------------------------------
def get_geographical_pos(G):

    pos = {}
    for node, data in G.nodes(data=True):
        if 'Latitude' in data and 'Longitude' in data:
            pos[node] = (data['Longitude'], data['Latitude'])

    return pos


# -------------------------------------------------------------
# Load deviation
# -------------------------------------------------------------
def compute_load_deviation(assign, loads):

    load_map = {}

    for s, c in assign.items():
        load_map[c] = load_map.get(c, 0) + loads[s]

    if not load_map:
        return 0

    return round(max(load_map.values()) - min(load_map.values()), 2)


# -------------------------------------------------------------
# Plot assignments
# -------------------------------------------------------------
def plot_assignments(G, pos, switches, controllers,
                     init_assign, final_assign,
                     loads, final_loads,
                     topology_name, save_dir,
                     controller_capacity,
                     extra_title="", file_tag=None):


    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(24, 12),      # poster friendly
        dpi=300,
        sharex=True,
        sharey=True
    )

    plt.subplots_adjust(wspace=0.05)

    for ax in (ax1, ax2):
        ax.axis('off')
        ax.set_aspect('equal')

    xs, ys = zip(*pos.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys))

    node_radius = span * 0.02


    # ---------------------------------------------------------
    # Controller colors
    # ---------------------------------------------------------

    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

    controller_colors = {
        c: color_list[i % len(color_list)]
        for i, c in enumerate(controllers)
    }

    migrated = {s for s in switches if init_assign[s] != final_assign[s]}


    # ---------------------------------------------------------
    # Compute loads
    # ---------------------------------------------------------

    init_loads_map = defaultdict(float)
    final_loads_map = defaultdict(float)

    for s, c in init_assign.items():
        init_loads_map[c] += loads[s]

    for s, c in final_assign.items():
        final_loads_map[c] += loads[s]


    # ---------------------------------------------------------
    # Draw assignment
    # ---------------------------------------------------------

    def draw_assignment(ax, assign, loads_map, highlight_migrations=False):

        controller_texts = []

        # ------------------------------
        # Draw edges with overload check
        # ------------------------------

        edge_colors = []
        edge_widths = []

        for u, v, data in G.edges(data=True):

            capacity = data.get("capacity", None)
            load = data.get("load", None)

            if capacity is not None and load is not None and load > delta * capacity:
                edge_colors.append("red")
                edge_widths.append(2.0)
            else:
                edge_colors.append("black")
                edge_widths.append(0.6)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            width=edge_widths
        )

        # ------------------------------
        # Draw switches
        # ------------------------------

        for n in switches:

            if n not in pos:
                continue

            x, y = pos[n]
            c = assign[n]

            if highlight_migrations and n in migrated:

                old_c = init_assign[n]
                new_c = final_assign[n]

                old_color = controller_colors[old_c]
                new_color = controller_colors[new_c]

                left_half = mpatches.Wedge(
                    (x, y), node_radius,
                    90, 270,
                    facecolor=old_color,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=3
                )

                right_half = mpatches.Wedge(
                    (x, y), node_radius,
                    270, 90,
                    facecolor=new_color,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=3
                )

                ax.add_patch(left_half)
                ax.add_patch(right_half)

            else:

                circle = mpatches.Circle(
                    (x, y),
                    node_radius,
                    facecolor=controller_colors[c],
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=3
                )

                ax.add_patch(circle)

            # switch label centered
            ax.text(
                x,
                y,
                str(n),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=5
            )


        # ------------------------------
        # Draw controllers
        # ------------------------------


        circle_diameter = 2 * node_radius
        square_side = 1.2 * circle_diameter

        for c in controllers:

            if c not in pos:
                continue

            x, y = pos[c]

            # square background
            square = mpatches.Rectangle(
                (x - square_side/2, y - square_side/2),
                square_side,
                square_side,
                facecolor=controller_colors[c],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.9,
                zorder=2
            )
            ax.add_patch(square)

            # circle on top
            circle = mpatches.Circle(
                (x, y),
                node_radius,
                facecolor=controller_colors[c],
                edgecolor='black',
                linewidth=1.2,
                zorder=3
            )
            ax.add_patch(circle)



            # Controller ID (fixed position)
            ax.text(
                x,
                y + square_side/2 + node_radius*0.4,
                f"C{c}",
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='bottom'
            )

            # Controller load (slightly higher)
            load_val = round(loads_map.get(c, 0), 1)

            cap_c = controller_capacity.get(c, 0) if isinstance(controller_capacity, dict) else controller_capacity
            threshold = delta * cap_c

            txt_color = 'red' if load_val > threshold else 'green'

            load_label = ax.text(
                x,
                y + square_side/2 + node_radius*1.2,
                f"{load_val}",
                fontsize=10,
                ha='center',
                va='bottom',
                color=txt_color
            )

            controller_texts.append(load_label)
           

        # prevent label overlaps
        adjust_text(
            controller_texts,
            ax=ax,
            expand_points=(1.02,1.05),
            force_text=0.05,
            only_move={'texts':'y'}
        )

    # ---------------------------------------------------------
    # Draw both plots
    # ---------------------------------------------------------

    draw_assignment(ax1, init_assign, init_loads_map, False)
    draw_assignment(ax2, final_assign, final_loads_map, True)


    # ---------------------------------------------------------
    # Titles
    # ---------------------------------------------------------

    load_dev_init = compute_load_deviation(init_assign, loads)
    load_dev_final = compute_load_deviation(final_assign, loads)

    ax1.set_title(f"Initial Association\nLoad Dev: {load_dev_init}", fontsize=20)
    ax2.set_title(f"Final Association\nLoad Dev: {load_dev_final}", fontsize=20)


    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    num_nodes = G.number_of_nodes()

    summary_handles = [
        Line2D([], [], linestyle='none', label=f"Nodes: {num_nodes} | Migrations: {len(migrated)}"),
        Line2D([], [], linestyle='none', label=f"Load deviation — init: {load_dev_init} | final: {load_dev_final}")
    ]

    icon_handles = [
        Line2D([0],[0], marker='s', linestyle='', color='w',
               markerfacecolor='lightgray', markeredgecolor='black',
               markersize=12, label='Controller (Square)'),

        Line2D([0],[0], marker='o', linestyle='', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=10, label='Switch (Circle)')
    ]

    controller_handles = []

    for c in controllers:

        cap_c = controller_capacity.get(c, 0) if isinstance(controller_capacity, dict) else controller_capacity
        usable = round(delta * cap_c,1)

        controller_handles.append(
            Line2D([0],[0],
                marker='s', linestyle='',
                markerfacecolor=controller_colors[c],
                markeredgecolor='black',
                markersize=12,
                label=f"C{c}: total={cap_c}, usable={usable}"
            )
        )

    legend_handles = summary_handles + icon_handles + controller_handles

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        fontsize=12,
        frameon=True,
        ncol=1
    )


    # ---------------------------------------------------------
    # Save figures
    # ---------------------------------------------------------

    tag = f"_{file_tag}" if file_tag else ""

    base_path = os.path.join(
        save_dir,
        f"{topology_name}{tag}_{num_nodes}nodes_{len(migrated)}migrations_{len(controllers)}controllers"
    )

    save_poster_figures(fig, base_path)

    plt.close(fig)

    return load_dev_init, load_dev_final