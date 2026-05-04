from equity_project.src.get_data import get_data
from equity_project.src.build_features import build_features
from equity_project.src.train import train
from equity_project.src.run_backtest import run_inference, run_backtest


def main():
    get_data()
    build_features()
    train()
    run_inference()
    run_backtest()


if __name__ == "__main__":
    main()
