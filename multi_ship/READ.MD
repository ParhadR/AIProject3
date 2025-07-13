# because of python file pathing just run these commands in your terminal to execute scripts

PYTHONPATH=. python multi_ship/generate_multi_ships.py
PYTHONPATH=. python multi_ship/combine_datasets.py
PYTHONPATH=. python multi_ship/train_multi_model.py
PYTHONPATH=. python multi_ship/run_multi_model.py