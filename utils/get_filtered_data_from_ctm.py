from utilities import (
    openctm,
    getbyFilter
)

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_ctm_file_path",
                        default="data/train/ctm",
                        type=str)

    parser.add_argument("--filter_condition",
                        default="-",
                        type=str)                      

    args = parser.parse_args()

    # variables
    input_ctm_file_path = args.input_ctm_file_path
    filter_condition = args.filter_condition

    # process
    words_or_phone_ctm = openctm(_ctm_file)
    words_or_phone_ctm_info = getbyFilter(
        words_or_phone_ctm,
        filter_condition
    )

    print(words_or_phone_ctm_info)
