from map_visualization import create_map
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Risk Map")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()

    lat = args.latitude
    lon = args.longitude

    create_map(lat, lon)


if __name__ == "__main__":
    main()