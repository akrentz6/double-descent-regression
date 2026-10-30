import argparse

from double_descent.experiments import add_common_arguments, run_and_save


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 0.01, 0.1, 1])
    args = parser.parse_args()
    settings = [(args.noise_std, alpha) for alpha in dict.fromkeys(args.alphas)]
    run_and_save(parser, args, settings, "ridge", group_by="alpha")


if __name__ == "__main__":
    main()
