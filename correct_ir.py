# %%

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def correct_attention_mask_names(path_to_ir: Path):
    tree = ET.parse(path_to_ir)
    root = tree.getroot()

    # %%
    for child in root[0]:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'attention_mask':
                for c in child:
                    print(c.tag)
                    if 'output' == c.tag:
                        for o in c:
                            if 'names' in o.attrib:
                                o.attrib['names'] = 'attention_mask'
                            print(o.attrib)

    # %%
    tree.write(path_to_ir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("path_to_ir", type=str, help="")
    args = parser.parse_args()

    # from pathlib import Path
    # for path in Path('/home/nlyaly/projects/lm-evaluation-harness/runs/').rglob('*all_eval_results.json'):
    #     if 'wiki' in str(path):
    #         parse_results(path)

    correct_attention_mask_names(Path(args.path_to_ir))




