#!/bin/bash

rm -rf build dist/* ripple_detector_CNN.egg-info
# python3 setup.py sdist bdist_wheel
# twine upload -r testpypi dist/*
pip uninstall ripple_detector_CNN -y
pip install --no-cache-dir --upgrade ./dist/ripple_detector_CNN-*-py3-none-any.whl --force-reinstall

## EOF
