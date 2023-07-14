#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file splits the training set into train and dev sets.
"""
import logging
from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def split_generic_train(prefix):
    src_dir = Path("data/manifests")

    manifests = read_manifests_if_cached(
        dataset_parts=["train", "dev", "test"],
        output_dir=src_dir,
        prefix=prefix,
        suffix="jsonl.gz",
        lazy=True,
    )
    assert manifests is not None

    # Handle the 'test' data
    print ("Generating test cuts...")
    test_cuts = CutSet.from_manifests(
        recordings=manifests["test"]["recordings"],
        supervisions=manifests["test"]["supervisions"]
    )
    test_cuts.to_file(src_dir / "cuts_test_raw.jsonl.gz")

    # Handle the 'dev' data
    print("Generating dev cuts...")
    dev_cuts = CutSet.from_manifests(
        recordings=manifests["dev"]["recordings"],
        supervisions=manifests["dev"]["supervisions"]
    )
    dev_cuts.to_file(src_dir / "cuts_dev_raw.jsonl.gz")
    
    # Handle the 'train' data
    print("Generating train cuts...")
    train_cuts = CutSet.from_manifests(
        recordings=manifests["train"]["recordings"],
        supervisions=manifests["train"]["supervisions"],
    )
    # Add speed perturbation
    train_cuts = (
        train_cuts + train_cuts.perturb_speed(0.9) + train_cuts.perturb_speed(1.1)
    )
    train_cuts.to_file(src_dir / "cuts_train_raw.jsonl.gz")



if __name__ == "__main__":
    import argparse
    import sys

    example = f"{sys.argv[0]} --prefix lang_generic"
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--prefix", "-p", help="Corpus prefix",
                        required=True)
    args = parser.parse_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    split_generic_train(args.prefix)
