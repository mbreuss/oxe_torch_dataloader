# minimum working example to load a single OXE dataset
from Uha import download_oxe_data, download_oxe_data_builder, make_pytorch_oxe_iterable_dataset


def main():
    dataloader = make_pytorch_oxe_iterable_dataset()


if __name__ == "__main__":
    main()
