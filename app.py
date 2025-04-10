import streamlit as st
from stmol import showmol
import py3Dmol
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from protein_analyzer import ProteinAnalyzer
from Bio.PDB import PDBParser
from ect import ECT, EmbeddedGraph, Directions
from pathlib import Path


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


def plot_3d_graph(coords, ax):
    """Plot 3D graph of protein backbone"""
    coords = np.array(coords)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "b-", linewidth=1.5, alpha=0.7)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="r", s=10, alpha=0.6)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])


def plot_ect_matrix(ect_result, phi, theta, threshold_idx, ax):
    """Plot ECT matrix visualization"""

    im = ax.imshow(
        ect_result, aspect="auto", extent=[0, 2 * np.pi, np.pi, 0], cmap="viridis"
    )
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Directions Index")
    ax.set_ylabel("Thresholds")
    ax.set_title("ECT Matrix")


def plot_ect_visualization(
    backbone_coords, ect_result, phi, theta, threshold_idx, protein_name
):
    """Create comprehensive ECT visualization"""
    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_subplot(131, projection="3d")
    plot_3d_graph(backbone_coords, ax1)
    ax1.set_title(f"Backbone Structure - {protein_name}")

    ax2 = fig.add_subplot(132, projection="3d")

    n_phi = len(phi)
    n_theta = len(theta)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    base_radius = abs(ect_result.thresholds[threshold_idx])
    ect_values = ect_result.reshape(-1, len(ect_result.thresholds))
    ect_values = ect_values[:, threshold_idx].reshape(n_theta, n_phi)

    perturbation_scale = (
        0.4 * base_radius / max(abs(np.min(ect_values)), abs(np.max(ect_values)))
    )
    radii = base_radius + perturbation_scale * ect_values

    x = radii * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii * np.cos(theta_grid)

    x_orig = base_radius * np.sin(theta_grid) * np.cos(phi_grid)
    y_orig = base_radius * np.sin(theta_grid) * np.sin(phi_grid)
    z_orig = base_radius * np.cos(theta_grid)

    max_abs_val = max(abs(np.min(ect_values)), abs(np.max(ect_values)))
    norm = plt.Normalize(np.min(ect_values), np.max(ect_values))

    surf = ax2.plot_surface(
        x, y, z, facecolors=plt.cm.viridis(norm(ect_values)), alpha=0.9
    )

    ax2.plot_wireframe(
        x_orig,
        y_orig,
        z_orig,
        color="black",
        alpha=0.2,
        linewidth=0.5,
        rstride=2,
        cstride=2,
    )

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
        ax=ax2,
        label=f"ECT Value at threshold {ect_result.thresholds[threshold_idx]:.2f}",
    )

    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("ECT Values as Sphere Perturbations")

    ax3 = fig.add_subplot(133)
    plot_ect_matrix(ect_result, phi, theta, threshold_idx, ax3)

    plt.tight_layout()
    return fig


def render_mol(pdb_file, style="backbone", single_chain=True):
    """Renders molecule in 3D using py3Dmol"""
    with open(pdb_file) as f:
        pdb_str = f.read()

    view = py3Dmol.view(width=None, height=500)
    view.removeAllModels()
    view.addModel(pdb_str, "pdb")

    if single_chain:
        analyzer = ProteinAnalyzer()
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]
        chain_id = analyzer.select_representative_chain(model)

        view.setStyle({"chain": "*"}, {"cartoon": {"style": "trace", "opacity": 0}})
        view.setStyle({"chain": "*", "atom": "*"}, {"sphere": {"radius": 0.0}})

        selector = {"chain": chain_id}
    else:
        selector = {}

    if style == "backbone":
        view.setStyle(selector | {"cartoon": {"style": "trace"}})
        view.addStyle(
            selector | {"atom": "C"}, {"sphere": {"radius": 0.3, "color": "spectrum"}}
        )
        view.addStyle(selector | {"bonds": True}, {"line": {"color": "spectrum"}})
    else:
        view.setStyle(selector | {"cartoon": {"color": "spectrum"}})

    view.zoomTo()
    return view


def plot_ect_streamlit(ect_result, backbone_coords, protein_name):
    """Creates side-by-side plot of backbone trace and ECT"""
    if not backbone_coords:
        st.warning("Cannot generate plots: No valid backbone coordinates found")
        return

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    coords = np.array(backbone_coords)
    ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], "b-", linewidth=2, alpha=0.7)
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="r", s=20, label="C atoms")

    ax1.set_title(f"Backbone Trace - {protein_name}", pad=10)
    ax1.set_xlabel("X (Å)")
    ax1.set_ylabel("Y (Å)")
    ax1.set_zlabel("Z (Å)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if ect_result is not None:
        ax2 = fig.add_subplot(122)
        ect_result.plot(ax=ax2)
        ax2.set_title(f"ECT Graph - {protein_name}", pad=10)
    else:
        ax2 = fig.add_subplot(122)
        ax2.text(
            0.5,
            0.5,
            "Could not calculate ECT",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax2.set_title("ECT Analysis", pad=10)

    plt.tight_layout()
    st.pyplot(fig)


def calculate_all_ects_and_project():
    """Calculate ECTs for all proteins and project them using PCA"""
    analyzer = ProteinAnalyzer()
    csv_dir = Path("csv")
    all_ects = []
    file_labels = []
    pdb_codes = []

    progress_text = "Calculating ECTs for all proteins..."
    progress_bar = st.progress(0)

    csv_files = list(csv_dir.glob("*.xlsx"))
    total_files = len(csv_files)

    for i, excel_file in enumerate(csv_files):
        df = pd.read_excel(excel_file)
        file_label = excel_file.stem

        if "File" not in df.columns:
            continue

        for pdb_code in df["File"].dropna():
            pdb_code = pdb_code.replace(".pdb", "").strip().lower()

            try:
                pdb_file = analyzer.download_pdb(pdb_code)
                if pdb_file is None:
                    continue

                backbone_coords = analyzer.get_backbone_coordinates(pdb_file)
                if not backbone_coords:
                    continue

                ect = analyzer.calculate_ect(backbone_coords)
                if ect is None:
                    continue

                ect_flat = ect.reshape(-1)
                all_ects.append(ect_flat)
                file_labels.append(file_label)
                pdb_codes.append(pdb_code)

            except Exception as e:
                st.warning(f"Error processing {pdb_code}: {str(e)}")

        progress_bar.progress((i + 1) / total_files)

    if not all_ects:
        st.error("No valid ECTs could be calculated")
        return

    X = np.array(all_ects)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = list(set(file_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_dict = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = [l == label for l in file_labels]
        points = X_pca[mask]
        c = color_dict[label]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=[c], label=label, alpha=0.6
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.legend()

    return fig, X_pca, pdb_codes, file_labels


def main():
    st.set_page_config(layout="wide", page_title="Protein Structure Viewer")

    st.markdown(
        """
        <style>
        .stmol-container { width: 100% !important; height: 100% !important; }
        iframe { min-width: 100% !important; width: 100% !important; overflow: hidden; }
        h2 a { color: inherit !important; text-decoration: none !important; }
        h2 a:hover { text-decoration: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧬 Protein Structure Analysis")

    analyzer = ProteinAnalyzer()

    st.sidebar.title("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:", ["Default Database", "CSV Files"]
    )

    if data_source == "Default Database":
        df = analyzer.load_protein_data()
    else:
        csv_dir = Path("csv")
        csv_files = list(csv_dir.glob("*.xlsx"))
        if not csv_files:
            st.error("No Excel files found in the csv directory")
            st.stop()

        selected_csv = st.sidebar.selectbox(
            "Select Excel file:", [f.stem for f in csv_files]
        )

        df = pd.read_excel(csv_dir / f"{selected_csv}.xlsx")

        if "File" not in df.columns:
            st.error(
                "Selected Excel file must have a 'File' column containing PDB codes"
            )
            st.stop()

        df["PDB Code"] = df["File"].apply(
            lambda x: x.replace(".pdb", "").strip().lower() if pd.notna(x) else x
        )

    st.sidebar.title("Select Protein")

    xray_only = st.sidebar.checkbox(
        "Show only X-ray structures",
        value=True,
        help="When enabled, only structures determined by X-ray crystallography will be shown",
    )

    available_proteins = []
    if data_source == "Default Database":
        if xray_only:
            for pdb_code in list(df[df["PDB Code"].notna()]["PDB Code"].unique()):
                pdb_file = analyzer.download_pdb(pdb_code)
                if pdb_file:
                    info = analyzer.get_structure_info(pdb_file)
                    if info["is_xray"]:
                        available_proteins.append(pdb_code)
        else:
            available_proteins = list(df[df["PDB Code"].notna()]["PDB Code"].unique())
    else:
        available_proteins = list(df["PDB Code"].dropna().unique())

    if len(available_proteins) == 0:
        st.error("No structures available with the current filters")
        st.stop()

    selected_protein = st.sidebar.selectbox("Choose a protein:", available_proteins)

    single_chain = st.sidebar.checkbox(
        "Show single chain only",
        value=True,
        help="When enabled, only the most representative chain will be shown for multi-chain proteins",
    )

    st.sidebar.title("ECT Parameters")
    num_directions = st.sidebar.slider(
        "Number of Directions",
        min_value=4,
        max_value=360,
        value=16,
        help="Number of directions to sample for ECT calculation. Higher values give more accurate results but take longer.",
    )

    num_thresholds = st.sidebar.slider(
        "Number of Thresholds",
        min_value=4,
        max_value=360,
        value=16,
        help="Number of distance thresholds to use. Higher values give finer detail but take longer.",
    )

    bound_radius = st.sidebar.slider(
        "Bounding Radius (Å)",
        min_value=1,
        max_value=360,
        value=16,
        help="Maximum distance to consider for ECT calculation. Should be large enough to capture the protein's structure.",
    )

    protein_info = df[df["PDB Code"] == selected_protein].iloc[0]

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown(
            f"""
            <h2>
                <a href="https://www.rcsb.org/structure/{selected_protein}" target="_blank">
                    Structure of {selected_protein} 🔗
                </a>
            </h2>
            """,
            unsafe_allow_html=True,
        )

        container = st.empty()
        pdb_file = analyzer.download_pdb(selected_protein)

        if pdb_file:
            with container:
                view = render_mol(pdb_file, single_chain=single_chain)
                if view:
                    showmol(view, height=500, width=None)
                else:
                    st.error("Failed to load protein structure")

            st.subheader("ECT")
            backbone_coords = analyzer.get_backbone_coordinates(
                pdb_file, single_chain=single_chain
            )
            if backbone_coords:
                n_phi = num_directions // 2
                n_theta = num_directions // 2
                vectors, phi, theta = create_direction_vectors(n_phi, n_theta)
                directions = Directions.from_vectors(vectors)

                ect = ECT(
                    directions=directions,
                    num_thresh=num_thresholds,
                )
                G = EmbeddedGraph()
                for i, coord in enumerate(backbone_coords):
                    G.add_node(node_id=i, coord=coord)
                    if i > 0:
                        G.add_edge(node_id1=i - 1, node_id2=i)
                radius = G.get_bounding_radius()
                ect_result = ect.calculate(G, override_bound_radius=radius)

                threshold_idx = num_thresholds // 2

                fig = plot_ect_visualization(
                    backbone_coords,
                    ect_result,
                    phi,
                    theta,
                    threshold_idx,
                    selected_protein,
                )
                st.pyplot(fig)
            else:
                st.error("""
                No valid backbone coordinates found.
                """)

    with col2:
        st.header("Protein Information")

        if data_source == "Default Database":
            st.subheader("Details")
            info_cols = {
                "Organism": "Organism",
                "Protein Name": "Protein",
                "Related to Parasitism": "Related to Parasitism",
                "Effector": "Effector",
            }

            for label, col in info_cols.items():
                st.write(f"**{label}:** {protein_info[col]}")

            st.subheader("Function")
            st.write(protein_info["Function"])
        else:
            with st.expander("CSV Data", expanded=False):
                for col in df.columns:
                    if col not in ["File", "PDB Code"]:
                        value = protein_info[col]
                        if pd.notna(value):
                            st.write(f"**{col}:** {value}")

        st.subheader("Structure Analysis")
        if pdb_file:
            stats = analyzer.analyze_structure(pdb_file, single_chain=single_chain)

            if single_chain:
                st.info(f"Showing data for chain {stats['selected_chain']}")

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Residues", stats["residues"])
                st.metric("Complete Residues", stats["complete_residue_count"])

            with metrics_col2:
                st.metric("Total Atoms", stats["atoms"])
                incomplete = stats["residues"] - stats["complete_residue_count"]
                if stats["sheet"] is not None:
                    st.metric("Sheet Residues", stats["sheet"])

            if stats["problematic_residues"]:
                with st.expander("⚠️ Structure Quality Information", expanded=True):
                    if stats["seqres_ranges"]:
                        chain_id = stats["selected_chain"] if single_chain else None
                        for chain, range_info in stats["seqres_ranges"].items():
                            if chain_id and chain != chain_id:
                                continue
                            st.write(f"**Chain {chain} Information:**")
                            st.write(
                                f"- Number of residues in structure: {range_info['actual_residues']}"
                            )
                            if (
                                range_info["seqres_length"]
                                != range_info["actual_residues"]
                            ):
                                st.write(
                                    f"- Expected sequence length: {range_info['seqres_length']} residues"
                                )
                            st.write(
                                f"- Residue numbering range: {range_info['start']} to {range_info['end']}"
                            )
                            st.write("")

                    st.write("**Residues with Missing Backbone Atoms:**")

                    missing_patterns = {}
                    for res_id, missing in stats["missing_atoms"].items():
                        if res_id not in stats.get("outside_range_residues", []):
                            key = ", ".join(sorted(missing))
                            if key not in missing_patterns:
                                missing_patterns[key] = []
                            missing_patterns[key].append(res_id)

                    if missing_patterns:
                        for missing_atoms, residues in missing_patterns.items():
                            st.markdown(f"Missing {missing_atoms}:")
                            sorted_residues = sorted(
                                residues,
                                key=lambda x: int("".join(filter(str.isdigit, x))),
                            )
                            groups = []
                            current_group = [sorted_residues[0]]
                            for res in sorted_residues[1:]:
                                prev_num = int(
                                    "".join(filter(str.isdigit, current_group[-1]))
                                )
                                curr_num = int("".join(filter(str.isdigit, res)))
                                if curr_num == prev_num + 1:
                                    current_group.append(res)
                                else:
                                    groups.append(current_group)
                                    current_group = [res]
                            groups.append(current_group)

                            for group in groups:
                                if len(group) > 2:
                                    st.write(f"- {group[0]} to {group[-1]}")
                                else:
                                    st.write(f"- {', '.join(group)}")
                    else:
                        st.write(
                            "No residues with missing atoms within the crystallized region."
                        )

                    if stats.get("outside_range_residues"):
                        st.write("")
                        st.write("**Note about Residue Numbering:**")
                        st.info("""
                        Some residue numbers in the PDB file are outside the crystallized construct. 
                        This is common when the residue numbers match the full protein sequence rather than 
                        just the crystallized fragment. These residues are not actually missing - they were 
                        not part of the experimental structure.
                        """)

        st.subheader("Visualization Options")
        style = st.selectbox(
            "Display style", ["backbone", "cartoon", "stick", "sphere", "line"]
        )

        color_scheme = st.selectbox(
            "Color scheme", ["spectrum", "chainid", "secondary structure", "residue"]
        )

        if st.button("Update View"):
            view = py3Dmol.view(width=None, height=500)
            view.removeAllModels()
            with open(pdb_file) as f:
                pdb_str = f.read()
            view.addModel(pdb_str, "pdb")

            color_map = {
                "spectrum": "spectrum",
                "chainid": "chainid",
                "secondary structure": "sstruc",
                "residue": "resname",
            }

            if single_chain:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("protein", pdb_file)
                model = structure[0]
                chain_id = analyzer.select_representative_chain(model)

                view.setStyle({"cartoon": {"style": "trace", "opacity": 0}})
                view.setStyle({"atom": "*"}, {"sphere": {"radius": 0.0}})

                selector = {"chain": chain_id}
            else:
                selector = {}

            if style == "backbone":
                view.setStyle(selector | {"cartoon": {"style": "trace"}})
                view.addStyle(
                    selector | {"atom": "C"},
                    {"sphere": {"radius": 0.3, "color": color_map[color_scheme]}},
                )
                view.addStyle(
                    selector | {"bonds": True},
                    {"line": {"color": color_map[color_scheme]}},
                )
            else:
                view.setStyle(selector | {style: {"color": color_map[color_scheme]}})

            view.zoomTo()

            with col1:
                container.empty()
                with container:
                    showmol(view, height=500, width=None)

    st.markdown("---")
    if st.button("Calculate ECTs and Show PCA Projection"):
        with st.spinner("Calculating ECTs and performing PCA analysis..."):
            fig, projections, codes, labels = calculate_all_ects_and_project()

            st.subheader("PCA Projection of ECT Matrices")
            st.pyplot(fig)

            df_proj = pd.DataFrame(
                {
                    "PDB Code": codes,
                    "Source File": labels,
                    "PC1": projections[:, 0],
                    "PC2": projections[:, 1],
                    "PC3": projections[:, 2],
                }
            )

            with st.expander("View Projection Data", expanded=False):
                st.dataframe(df_proj)

    st.markdown("---")
    with st.expander("📋 Detailed PDB Information", expanded=False):
        if pdb_file:
            pdb_info = analyzer.get_structure_info(pdb_file, single_chain=single_chain)

            if single_chain:
                st.info(f"Showing data for chain {pdb_info['selected_chain']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Structure Information")
                st.write(f"**Resolution:** {pdb_info['resolution']} Å")
                st.write(f"**Method:** {pdb_info['structure_method']}")
                st.write(f"**Deposition Date:** {pdb_info['deposition_date']}")

            with col2:
                st.subheader("Chain Information")
                chain_data = []
                for chain in pdb_info["chains"]:
                    chain_data.append(
                        {
                            "Chain ID": chain["chain_id"],
                            "Residues": chain["residue_count"],
                            "Atoms": chain["atom_count"],
                        }
                    )
                chain_df = pd.DataFrame(chain_data)
                st.dataframe(chain_df, use_container_width=True)

            with col3:
                st.subheader("Composition")
                total_residues = sum(
                    chain["residue_count"] for chain in pdb_info["chains"]
                )
                total_atoms = sum(chain["atom_count"] for chain in pdb_info["chains"])
                st.metric("Total Atoms", total_atoms)
                if stats["helix"] is not None:
                    helix_percent = round(stats["helix"] / total_residues * 100, 1)
                    sheet_percent = round(stats["sheet"] / total_residues * 100, 1)
                    st.metric("Helix Content", f"{helix_percent}%")
                    st.metric("Sheet Content", f"{sheet_percent}%")

            st.subheader("Raw PDB Header Information")
            header_df = pd.DataFrame(
                [{k: str(v) for k, v in pdb_info["raw_header"].items()}]
            )
            st.dataframe(header_df.T, use_container_width=True)


if __name__ == "__main__":
    main()
