"""Quick run with smaller data."""
import config
config.NUM_SYNTHETIC_PAIRS = 5000
config.NUM_EVAL_QUERIES = 100
config.DEFAULT_EPOCHS = 3
config.DEFAULT_HARD_NEGS = 2
config.DEFAULT_TOP_K = 20
config.EVAL_K_VALUES = [1, 5, 10, 20]

from run_experiments import main
main()
