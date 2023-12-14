import numpy as np
from collections import defaultdict


char_to_idx = {
    '.': -1,
    '|': 0,
    '-': 1,
    '/': 2,
    '\\': 3,
}

def _idx_to_char(idx):
    for (k, v) in char_to_idx.items():
        if v == idx:
            return k
    raise ValueError

"""Converts a human-readable block of text to an input for our model"""
def ascii_to_input(text):
    result = np.zeros((8,8,4))
    lines = text.strip().split("\n")
    assert(len(lines) == 8)

    for i, line in enumerate(lines):
        assert(len(line) == 8)
        for j, c in enumerate(line):
            index = char_to_idx[c]
            if index >= 0:
                result[i][j][index] = 1
    return result


def weights_to_ascii(arr):
    lines = []
    for row in arr:
        line = []
        for vals in row:
            if (vals == 0).all():
                line.append(".")
            else:
                idx = np.argmax(vals)
                line.append(_idx_to_char(idx))
        lines.append(''.join(line))
    return "\n".join(lines)


def load_inputs_from_file(filename):
    with open(filename, "r") as openfile:
        text = openfile.read()

    lines = text.split("\n")
    sequences = defaultdict(list)
    i = 0

    while i < len(lines):
        if lines[i].strip() == "":
            continue
        # Read which sequence this example is part of
        seq_no = int(lines[i])
        # Next 8 lines contain image
        data = "\n".join(lines[(i + 1) : (i + 9)])
        data = ascii_to_input(data)
        sequences[seq_no].append(data)

        i += 9

    # Return list of sequences without ids
    return list(sequences.values())



# Here's an example of the format:
_example_text = """
../.....
./......
/.......
........
........
........
........
........
"""
