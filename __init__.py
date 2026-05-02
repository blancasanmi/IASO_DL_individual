import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app_ import app, image, volume, VOLUME_PATH, TENSORS_PATH
from preprocessing_app import prepare_data
from train_app import run_tuning
from evaluate_app import final_evaluation