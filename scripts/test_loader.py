# minimum working example to load a single OXE dataset
from Uha import download_oxe_data, download_oxe_data_builder, make_pytorch_oxe_iterable_dataset
import tqdm


def main():
    dataloader = make_pytorch_oxe_iterable_dataset()
    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print(sample)
        if i == 5000:
            break


if __name__ == "__main__":
    main()
