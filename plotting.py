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


def plot_final_vs_recovery_assignment(
    G, pos, switches, controllers,
    final_assign,
    recovery_assign,
    loads,
    final_loads,
    recovery_loads,
    topology_name,
    save_dir,
    controller_capacity,
    failed_controller,
    backup_controller,
    backup_capacity=None,
    backup_pos=None,
    file_tag=None,
):
    os.makedirs(save_dir, exist_ok=True)

    pos2 = dict(pos)

    if backup_controller is not None and backup_controller not in pos2:
        affected = [s for s in switches if recovery_assign.get(s) == backup_controller and s in pos2]
        if affected:
            xs = [pos2[s][0] for s in affected]
            ys = [pos2[s][1] for s in affected]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            xs0, ys0 = zip(*pos2.values())
            cx, cy = max(xs0), sum(ys0) / len(ys0)

        xs0, ys0 = zip(*pos2.values())
        span0 = max(max(xs0) - min(xs0), max(ys0) - min(ys0))
        pos2[backup_controller] = backup_pos if backup_pos else (cx + 0.12 * span0, cy)

    original_controllers = list(dict.fromkeys(controllers))

    active_recovery_controllers = [
        c for c in original_controllers
        if c != failed_controller
    ]

    if backup_controller is not None and backup_controller not in active_recovery_controllers:
        active_recovery_controllers.append(backup_controller)

    all_plot_controllers = list(dict.fromkeys(original_controllers + active_recovery_controllers))

    if backup_capacity is not None and backup_controller is not None:
        controller_capacity = dict(controller_capacity)
        controller_capacity[backup_controller] = backup_capacity

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(24, 12),
        dpi=300,
        sharex=True,
        sharey=True
    )
    plt.subplots_adjust(wspace=0.05)

    for ax in (ax1, ax2):
        ax.axis("off")
        ax.set_aspect("equal")

    xs, ys = zip(*pos2.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    node_radius = span * 0.02

    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    controller_colors = {
        c: color_list[i % len(color_list)]
        for i, c in enumerate(all_plot_controllers)
    }

    failed_orphan_switches = {
        s for s in switches
        if final_assign.get(s) == failed_controller
    }

    planned_switches = {
        s for s in switches
        if final_assign.get(s) == failed_controller
        and backup_controller is not None
        and recovery_assign.get(s) == backup_controller
    }

    def draw_failed_controller_box(ax, square_side):
        if failed_controller not in pos2:
            return

        x, y = pos2[failed_controller]

        failed_box = mpatches.Rectangle(
            (x - square_side / 2, y - square_side / 2),
            square_side,
            square_side,
            facecolor="white",
            edgecolor="red",
            linewidth=1.8,
            linestyle="--",
            zorder=2
        )
        ax.add_patch(failed_box)

        ax.text(
            x,
            y + square_side / 2 + node_radius * 1.3,
            "FAILED",
            fontsize=10,
            fontweight="bold",
            color="red",
            ha="center",
            va="bottom",
            zorder=8
        )

    def draw_one(ax, assign, loads_map, active_ctrls, is_recovery=False):
        controller_texts = []

        nx.draw_networkx_edges(
            G, pos2, ax=ax,
            edge_color="black",
            width=0.6
        )

        if is_recovery and backup_controller is not None:
            bx, by = pos2[backup_controller]
            for s in planned_switches:
                if s in pos2:
                    sx, sy = pos2[s]
                    ax.plot(
                        [sx, bx], [sy, by],
                        linestyle="--",
                        linewidth=1.4,
                        color=controller_colors[backup_controller],
                        zorder=1
                    )

        for n in switches:
            if n not in pos2:
                continue

            x, y = pos2[n]
            c = assign.get(n, final_assign.get(n))

            if (not is_recovery) and n in failed_orphan_switches:
                face = "white"
            else:
                face = controller_colors.get(c, "lightgray")

            circle = mpatches.Circle(
                (x, y),
                node_radius,
                facecolor=face,
                edgecolor="black",
                linewidth=1.2,
                zorder=3
            )
            ax.add_patch(circle)

            ax.text(
                x, y, str(n),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5
            )

        circle_diameter = 2 * node_radius
        square_side = 1.2 * circle_diameter

        for c in active_ctrls:
            if c not in pos2:
                continue

            if (not is_recovery) and c == failed_controller:
                continue

            x, y = pos2[c]

            square = mpatches.Rectangle(
                (x - square_side / 2, y - square_side / 2),
                square_side,
                square_side,
                facecolor=controller_colors[c],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.9,
                zorder=2
            )
            ax.add_patch(square)

            circle = mpatches.Circle(
                (x, y),
                node_radius,
                facecolor=controller_colors[c],
                edgecolor="black",
                linewidth=1.2,
                zorder=3
            )
            ax.add_patch(circle)

            ax.text(
                x,
                y + square_side / 2 + node_radius * 0.4,
                f"C{c}",
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="bottom"
            )

            load_val = round(loads_map.get(c, 0), 1)
            cap_c = controller_capacity.get(c, 0) if isinstance(controller_capacity, dict) else controller_capacity
            threshold = delta * cap_c
            txt_color = "red" if load_val > threshold else "green"

            load_label = ax.text(
                x,
                y + square_side / 2 + node_radius * 1.2,
                f"{load_val}",
                fontsize=10,
                ha="center",
                va="bottom",
                color=txt_color,
                fontweight="bold"
            )
            controller_texts.append(load_label)

        if not is_recovery:
            draw_failed_controller_box(ax, square_side)

        if is_recovery:
            draw_failed_controller_box(ax, square_side)

        adjust_text(
            controller_texts,
            ax=ax,
            expand_points=(1.02, 1.05),
            force_text=0.05,
            only_move={"texts": "y"}
        )

    load_dev_final = compute_load_deviation(final_assign, loads)
    load_dev_recovery = compute_load_deviation(recovery_assign, loads)

    draw_one(ax1, final_assign, final_loads, original_controllers, is_recovery=False)
    draw_one(ax2, recovery_assign, recovery_loads, active_recovery_controllers, is_recovery=True)

    ax1.set_title(f"Final Association\nLoad Dev: {load_dev_final}", fontsize=20)
    ax2.set_title(
        f"Planned Recovery Association Failure of C{failed_controller}\n"
        f"Load Dev: {load_dev_recovery}",
        fontsize=20
    )

    summary_handles = [
        Line2D([], [], linestyle="none",
               label=f"Nodes: {G.number_of_nodes()} | Failed Controller: C{failed_controller}"),
        Line2D([], [], linestyle="none",
               label=f"Load deviation — final: {load_dev_final} | recovery: {load_dev_recovery}")
    ]

    icon_handles = [
        Line2D([0], [0], marker="s", linestyle="", color="w",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=12, label="Controller (Square)"),
        Line2D([0], [0], marker="o", linestyle="", color="w",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label="Switch (Circle)"),
        Line2D([0], [0], marker="s", linestyle="--", color="red",
               markerfacecolor="white", markeredgecolor="red",
               markersize=12, label=f"Failed C{failed_controller}")
    ]

    if backup_controller is not None:
        icon_handles.append(
            Line2D([0], [0], linestyle="--",
                   color=controller_colors[backup_controller],
                   linewidth=1.4,
                   label=f"Planned links to C{backup_controller}")
        )

    controller_handles = []
    seen_legend_ctrls = set()

    for c in active_recovery_controllers:
        if c in seen_legend_ctrls:
            continue
        seen_legend_ctrls.add(c)

        cap_c = controller_capacity.get(c, 0) if isinstance(controller_capacity, dict) else controller_capacity
        usable = round(delta * cap_c, 1)

        suffix = " New Backup" if c == backup_controller else ""

        controller_handles.append(
            Line2D([0], [0],
                   marker="s",
                   linestyle="",
                   markerfacecolor=controller_colors[c],
                   markeredgecolor="black",
                   markersize=12,
                   label=f"C{c}{suffix}: total={round(cap_c,1)}, usable={usable}")
        )

    fig.legend(
        handles=summary_handles + icon_handles + controller_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        fontsize=12,
        frameon=True,
        ncol=1
    )

    tag = f"_{file_tag}" if file_tag else ""
    base_path = os.path.join(
        save_dir,
        f"{topology_name}{tag}_failC{failed_controller}_backupC{backup_controller}"
    )

    save_poster_figures(fig, base_path)
    plt.close(fig)

    return load_dev_final, load_dev_recovery