# minimum working example to load a single OXE dataset
from Uha import download_oxe_data, download_oxe_data_builder


def main():
    download_oxe_data_builder("~/tensorflow_datasets")


if __name__ == "__main__":
    main()
