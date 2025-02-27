import re

text = """broadleaved_indigenous_hardwood     BIH
deciduous_hardwood                  DHW
grose_broom                         GBM
harvested_forest                    HFT
herbaceous_freshwater_vege          HFV
high_producing_grassland            HPG
indigenous_forest                   IFT
lake_pond                           LPD
low_producing_grassland             LPG
manuka_kanuka                       MKA
shortrotation_cropland              SCL
urban_build_up                      UBU
urban_parkland                      UPL"""

lines = text.split("\n")
values = [re.split(r"\s+", l.strip()) for l in lines] #[l.split(maxsplit=1) for l in lines]
classes = {v[0]: v[1] for v in values}

import json
print(json.dumps(classes, indent=2))

if len(set(classes.values())) == len(classes.values()):
    print("CLASSES ARE UNIQUE")
