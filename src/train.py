from utils.utils import (
    get_args,
    set_seed
)
from utils.dataset_utils import (
    split_dataset
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from loader.make import (
    make_loader
)
def main():
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)
    set_seed(config.seed)
    dataset_split_dict = split_dataset(config.dirs.data_dir)
    loader = make_loader(config, dataset_split_dict)
    print("Data Loader Created")
    for batch in loader:
        print(batch)
        break
    

if __name__ == '__main__':
    main()
