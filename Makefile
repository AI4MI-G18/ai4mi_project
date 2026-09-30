red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
## Number of processes for the slicing (-1: all cores), e.g. `make data/SEGTHOR PROCS=8`
PROCS ?= -1

## Full training set (40 patients)
data/segthor_train: data/segthor_train_full.zip
	$(info $(yellow)unzip $<$(reset))
#	sha256sum -c data/segthor_train_full.sha256
	rm -rf $@_tmp $@
	unzip -q $< -d $@_tmp
	rm -f $@_tmp/.DS_STORE $@_tmp/train/.DS_STORE
	mv $@_tmp $@

data/SEGTHOR: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 10 -p $(PROCS)
	mv $@_tmp $@

## Same split (same --seed/--retain), with the preprocessing of preprocessing.py
## [-500, 500] HU keeps the soft tissue organs (heart, aorta, esophagus) and the
## air/tissue edge of the trachea, while removing the metal/contrast outliers (up to 31743 HU)
## 1.5 mm in-plane gives a 384 mm field of view at 256x256, organs being within 147 mm of the center
HU_WINDOW := --hu_window -500 500
SPACING := --spacing 1.5 1.5

data/SEGTHOR_HU: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 10 -p $(PROCS) $(HU_WINDOW)
	mv $@_tmp $@

data/SEGTHOR_RESAMPLE: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 10 -p $(PROCS) $(SPACING)
	mv $@_tmp $@

data/SEGTHOR_PREPROC: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256 --retain 10 -p $(PROCS) $(HU_WINDOW) $(SPACING)
	mv $@_tmp $@
