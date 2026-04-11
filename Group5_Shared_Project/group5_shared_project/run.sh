#!/usr/bin/env bash
python src/data_loader.py --subset_size 2000
python src/supervised_experiments.py
