# Registering a loader

To read a custom input format, implement a `BaseLoader` and register it with
`LoaderRegistry`. Its `sniff` method recognises the format from the first lines
of a file; `load` parses the file into a `Measurement`.

This example writes an invented CSV file with a format marker and three familiar
quantity columns. The loader recognises the marker, reads the table and records
the format name on the returned measurement.

```python
from pathlib import Path

import pandas as pd

import calamari
from calamari.io import BaseLoader, LoaderRegistry, Measurement

Path("myrig.csv").write_text(
    "# myrig v1\nTemperature (K),Magnetic Field (Oe),Moment (emu)\n5,1000,0.001\n"
)


@LoaderRegistry.register
class MyRigLoader(BaseLoader):
    format_name = "MyRig"

    @classmethod
    def sniff(cls, lines):
        return bool(lines) and lines[0].startswith("# myrig v")

    @classmethod
    def load(cls, path):
        data = pd.read_csv(path, comment="#")
        return Measurement(path=Path(path), data_format=cls.format_name,
                           measurement_type="unknown", file_role="summary",
                           title=None, data=data)


measurement = calamari.load("myrig.csv")
print(measurement.data_format)
print(calamari.sniff_format("myrig.csv"))
print(measurement.field_oe.tolist())
print(measurement.moment_emu.tolist())
```

<!-- output -->
```text
MyRig
MyRig
[1000]
[0.001]
```

The column names and unit suffixes let the existing field and moment accessors
resolve these quantities. Other accessors and analyses require their own quantities
and metadata; registration alone does not establish compatibility with every workflow.
`find_files`, `sniff_format` and `is_supported_file` use the same loader registry.

The [loader API](api/calamari.io.base) defines the interface. For a maintained loader,
validate recognised file structure, document its units and evidence, and add synthetic
regressions for supported file shapes. Follow the [architecture contracts](ARCHITECTURE.md)
and [loading contracts](USER_GUIDE.md) when extending the built-in MPMS3 support.
