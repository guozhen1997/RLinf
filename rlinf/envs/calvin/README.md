# CALVIN Environment Setup Guide

## Installation
1. Git clone the CALVIN environment with:
```
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
```

2. Install with:
```
cd $CALVIN_ROOT
sh install.sh
```

3. Download a small debug dataset (1.3 GB) for evaluation:
```
cd $CALVIN_ROOT/dataset
sh download_data.sh debug
```

4. Replace the dataset path below, which follows the official evaluation pipeline in [calvin/eval](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py):
```
def make_env():
    dataset_paths = [
        "calvin_debug_dataset/",
    ]
    for path in dataset_paths:
        try:
            return get_env(Path(path) / "validation", show_gui=False)
        except Exception:
            continue
    raise RuntimeError(f"Please download the calvin_debug_dataset from https://github.com/mees/calvin#computer--quick-start with `sh download_data.sh debug`.")
```

## Getting Help
If you encounter issues not addressed in this guide, please:
1. Refer to the [CALVIN documentation](https://github.com/mees/calvin)
2. Create an issue in the RLinf repository

## License
Please refer to the individual repository licenses for the CALVIN.
