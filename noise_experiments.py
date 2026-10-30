import argparse

from double_descent.experiments import add_common_arguments, run_and_save


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0, 0.5, 1])
    args = parser.parse_args()
    settings = [(sigma, 0.0) for sigma in dict.fromkeys(args.noise_levels)]
    run_and_save(parser, args, settings, "noise", group_by="noise_std")


if __name__ == "__main__":
    main()
