# Group 5 Shared Project 

This package provides a clean shared base for the Histopathologic Cancer Detection group project.

## Folder structure
- `src/` shared code and experiment entry points
- `results/screenshots/` EDA screenshots
- `results/graphs/` training curves
- `results/logs/` CSV logs
- `results/models/` saved models
- `results/tables/` split files and comparison tables
- `report/` report support files

## Recommended workflow
1. Place `train_labels.csv` in `data/`
2. Place training `.tif` images in `data/train/`
3. Run:
   - `python src/data_loader.py --subset_size 2000`
4. Then run CNN experiments:
   - `python src/supervised_experiments.py`

## Notes
- Everyone should use the same split CSV files from `results/tables/`
- This package is structured to support data preparation, CNN training, transfer learning, and SOTA experiments
