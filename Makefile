.PHONY: docs
all: utils/feature_matching.pyx
	cd utils; python setup.py build_ext --inplace
docs:
	cd utils; python setup.py build_ext --inplace
	cd docs; make html
