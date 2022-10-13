# --- Just Revisiting Deep Learning Models for Tabular Data

# if you change model, check, run.yaml

# -- train default [Single]
python main.py --action train --single 1

# -- train fold [Ensemble]
python main.py --action train --single 0


# -- test default [Single]
python main.py --action test --single 1


# -- test fold [Ensemble]
python main.py --action test --single 0

# --- run.yaml

            
# train : 