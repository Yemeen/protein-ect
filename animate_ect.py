import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from protein_analyzer import ProteinAnalyzer
from ect import ECT, EmbeddedGraph, Directions
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pandas as pd
import traceback


def create_direction_vectors(n_phi, n_theta):
    """Create direction vectors for ECT calculation"""
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)

    vectors = []
    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            vectors.append([x, y, z])

    return np.array(vectors), phi, theta


def plot_ect_sphere(
    ect_result, phi, theta, threshold_idx, fig, ax, global_min, global_max, pdb_code
):
    """Plot ECT sphere for a given threshold"""
    n_phi = len(phi)
    n_theta = len(theta)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    base_radius = 1.0
    ect_values = ect_result.reshape(-1, len(ect_result.thresholds))
    ect_values = ect_values[:, threshold_idx].reshape(n_theta, n_phi)

    perturbation_scale = 0.3
    normalized_values = (ect_values - global_min) / (global_max - global_min) - 0.5
    radii = base_radius * (1 + perturbation_scale * normalized_values)

    x = radii * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii * np.cos(theta_grid)

    x_orig = base_radius * np.sin(theta_grid) * np.cos(phi_grid)
    y_orig = base_radius * np.sin(theta_grid) * np.sin(phi_grid)
    z_orig = base_radius * np.cos(theta_grid)

    norm = plt.Normalize(global_min, global_max)

    ax.clear()

    surf = ax.plot_surface(
        x, y, z, facecolors=plt.cm.viridis(norm(ect_values)), alpha=0.9
    )

    ax.plot_wireframe(
        x_orig,
        y_orig,
        z_orig,
        color="black",
        alpha=0.2,
        linewidth=0.5,
        rstride=2,
        cstride=2,
    )

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"ECT Sphere at threshold {ect_result.thresholds[threshold_idx]:.2f} for {pdb_code}"
    )

    ax.set_box_aspect([1, 1, 1])

    return surf


def animate_ect_sphere(pdb_code, output_dir="animations", skip_if_exists=False):
    """Generate animation of ECT sphere across thresholds"""

    Path(output_dir).mkdir(exist_ok=True)

    output_path = Path(output_dir) / f"{pdb_code}_ect_animation.gif"
    if skip_if_exists and output_path.exists():
        print(f"Animation for {pdb_code} already exists, skipping...")
        return True

    try:
        analyzer = ProteinAnalyzer()
        pdb_file = analyzer.download_pdb(pdb_code)
        if not pdb_file:
            print(f"Could not download PDB file for {pdb_code}")
            return False

        backbone_coords = analyzer.get_backbone_coordinates(pdb_file, single_chain=True)

        if not backbone_coords:
            print(f"No valid backbone coordinates found for {pdb_code}")
            return False

        n_phi = 150
        n_theta = 150
        vectors, phi, theta = create_direction_vectors(n_phi, n_theta)
        directions = Directions.from_vectors(vectors)

        num_thresholds = 60
        frames_per_threshold = 3
        total_frames = num_thresholds * frames_per_threshold

        ect = ECT(directions=directions, num_thresh=num_thresholds)

        G = EmbeddedGraph()
        for i, coord in enumerate(backbone_coords):
            G.add_node(node_id=i, coord=coord)
            if i > 0:
                G.add_edge(node_id1=i - 1, node_id2=i)

        ect_result = ect.calculate(G)

        all_values = ect_result.reshape(-1, len(ect_result.thresholds))
        global_min = np.min(all_values)
        global_max = np.max(all_values)

        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = GridSpec(1, 2, width_ratios=[4, 0.2], figure=fig)

        ax = fig.add_subplot(gs[0, 0], projection="3d")

        cax = fig.add_subplot(gs[0, 1])

        norm = plt.Normalize(global_min, global_max)
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
            cax=cax,
            label="ECT Value",
        )

        def update(frame):
            base_idx = frame // frames_per_threshold
            frac = (frame % frames_per_threshold) / frames_per_threshold

            if base_idx >= num_thresholds - 1:
                threshold_idx = num_thresholds - 1
            else:
                threshold_idx = base_idx + frac

            surf = plot_ect_sphere(
                ect_result,
                phi,
                theta,
                int(threshold_idx),
                fig,
                ax,
                global_min,
                global_max,
                pdb_code,
            )

            ax.view_init(elev=20, azim=frame * (360.0 / total_frames))
            return (surf,)

        anim = FuncAnimation(
            fig, update, frames=total_frames, interval=33.33, blit=True
        )

        anim.save(str(output_path), writer="pillow", fps=30, dpi=100)
        plt.close()

        print(f"Successfully created animation for {pdb_code}")
        return True

    except Exception as e:
        print(f"Error processing {pdb_code}:")
        print(traceback.format_exc())
        plt.close()
        return False


def process_all_pdbs(output_dir="animations"):
    """Process all PDB files in the dataset"""

    analyzer = ProteinAnalyzer()
    df = analyzer.load_protein_data()

    pdb_codes = df[df["PDB Code"].notna()]["PDB Code"].unique()

    print(f"Found {len(pdb_codes)} PDB codes to process")

    results = []
    for i, pdb_code in enumerate(pdb_codes, 1):
        print(f"\nProcessing {pdb_code} ({i}/{len(pdb_codes)})...")
        success = animate_ect_sphere(pdb_code, output_dir)
        results.append({"pdb_code": pdb_code, "success": success})

    summary_df = pd.DataFrame(results)
    success_count = summary_df["success"].sum()
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(pdb_codes)}")

    failed_cases = summary_df[~summary_df["success"]]
    if not failed_cases.empty:
        failed_file = Path(output_dir) / "failed_cases.csv"
        failed_cases.to_csv(failed_file, index=False)
        print(f"Failed cases saved to: {failed_file}")


if __name__ == "__main__":
    process_all_pdbs()
