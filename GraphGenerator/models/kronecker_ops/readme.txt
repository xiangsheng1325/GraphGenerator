# readme before using kronecker graph generator
In order to reduce space, we compressed the source codes of kronecker graph generator.
Before using kronecker, we need to decompress and compile them:

1. unzip `kronecker_src.zip`

```bash
cd path_to/GraphGenerator/GraphGenerator/models/kronecker_ops
unzip -o -d . kronecker_src.zip
```

2. reinstall this package
```bash
pip uninstall GraphGenerator
cd path_to/GraphGenerator
pip install -e .
```

