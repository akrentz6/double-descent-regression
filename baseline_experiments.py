import argparse

from double_descent.experiments import add_common_arguments, run_and_save


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--noise-std", type=float, default=0.5)
    args = parser.parse_args()
    run_and_save(parser, args, [(args.noise_std, 0.0)], "baseline")


if __name__ == "__main__":
    main()
