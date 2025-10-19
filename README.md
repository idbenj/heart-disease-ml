# Heart Disease ML (EPFL CS-233)

Logistic Regression, k-NN, and K-Means implemented from scratch (NumPy/Matplotlib) for the Heart Disease dataset (5-class classification).  
No scikit-learn used unless allowed; code structured as classes under `src/methods/`.

## Structure
- `main.py`: run selected method and evaluation
- `test_ms1.py`, `test_ms2.py`: sanity tests provided for milestones
- `src/`: data loading, utilities, and method classes
  - `methods/`: `logistic_regression.py`, `knn.py`, `kmeans.py`, `deep_network.py`, `dummy_methods.py`
- `report.pdf`: project write-up

## Run
```bash
python main.py --data_path <path_to_data> --method logistic_regression --lr 1e-5 --max_iters 100
python main.py --data_path <path_to_data> --method knn --K 3
python main.py --data_path <path_to_data> --method kmeans --K 5
