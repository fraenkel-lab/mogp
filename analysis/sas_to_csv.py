#!/usr/bin/env python3

from sas7bdat import SAS7BDAT
from pathlib import Path

def sas_to_csv(in_path, save_path):
    """Light helper function to convert sas files to csv"""
    sas_list = list(in_path.glob('*.sas7bdat'))
    assert (len(sas_list) > 0), 'Input directory does not have any sas7bdat files'
    Path.mkdir(save_path, parents=True, exist_ok=True)

    for sas in sas_list:
        print('Converting: ', sas.stem)
        df = SAS7BDAT(str(sas)).to_data_frame()
        df.to_csv(save_path / '{}.csv'.format(sas.stem))


if __name__ == "__main__":
    # Used for ceftriaxone data; originally in sas7bdat format
    ceft_in_path = Path('data/raw_data/ceft')
    ceft_save_path = Path('data/processed_data/ceft')
    sas_to_csv(ceft_in_path, ceft_save_path)
