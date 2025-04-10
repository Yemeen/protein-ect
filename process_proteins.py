from pathlib import Path
import pandas as pd
from protein_analyzer import ProteinAnalyzer
import matplotlib.pyplot as plt


def process_csv_files():
    """Process all Excel files in the csv directory and calculate ECT for each protein"""
    csv_dir = Path("csv")
    protein_analyzer = ProteinAnalyzer(pdb_dir="pdb_files")

    for excel_file in csv_dir.glob("*.xlsx"):
        try:
            protein_type = excel_file.stem.split("_")[-1]

            output_dir = Path("protein_plots") / protein_type
            output_dir.mkdir(exist_ok=True)

            df = pd.read_excel(excel_file)

            for pdb_code in df["File"].dropna():
                pdb_code = pdb_code.replace(".pdb", "").strip().lower()

                print(f"Processing {pdb_code} for {protein_type}")

                pdb_file = protein_analyzer.download_pdb(pdb_code)
                if pdb_file is None:
                    print(f"Failed to download {pdb_code}")
                    continue

                backbone_coords = protein_analyzer.get_backbone_coordinates(pdb_file)
                if not backbone_coords:
                    print(f"No backbone coordinates found for {pdb_code}")
                    continue

                ect = protein_analyzer.calculate_ect(backbone_coords)
                if ect is None:
                    print(f"Failed to calculate ECT for {pdb_code}")
                    continue

                plot_filename = output_dir / f"{pdb_code}_ect.png"

                plt.figure(figsize=(10, 8))
                ect.plot()
                plt.savefig(plot_filename)
                plt.close()

                print(f"Saved ECT plot to {plot_filename}")

        except Exception as e:
            print(f"Error processing {excel_file}: {str(e)}")


if __name__ == "__main__":
    process_csv_files()
