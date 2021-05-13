################################################################################
## Singularity image file version
################################################################################
export SINGULARITY_SIF_ROOT=$PJ_HOME/singularity/${SINGULARITY_FNAME}
export SINGULARITY_SIF_PATH=$PJ_HOME/singularity/${SINGULARITY_FNAME}.sif


################################################################################
## Singularity image file version
################################################################################
# singularity shell
function sshell() {
    echo singularity shell --nv $SINGULARITY_SIF_PATH
    singularity shell --nv $SINGULARITY_SIF_PATH
}

# singularity python
function spy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH python3 $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH python3 $@
}

# singularity ipython
function sipy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH ipython $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH ipython $@
}

# singularity jupyter
function sjpy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH jupyter-notebook $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH jupyter-notebook $@
}

# singularity build
function sbuild () {
    DEF=$1
    OPTION=$2

    FNAME=`echo $DEF | cut -d . -f 1`
    SIF=${FNAME}.sif

    echo singularity build $OPTION $SIF $DEF
    singularity build $OPTION $SIF $DEF
    
}

################################################################################
## Sandbox version
################################################################################
# writable singularity shell
function sshellw() {
    echo singularity shell --nv $SINGULARITY_SIF_ROOT
    singularity shell --nv $SINGULARITY_SIF_ROOT
}

# writable singularity python
function spyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT python3 $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT python3 $@
}

# writable singularity ipython
function sipyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT ipython $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT ipython $@
}

# writable singularity jupyter-notebook
function sjpyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT jupyter-notebook $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT jupyter-notebook $@
}

# writable singularity build
function sbuildw () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --fakeroot --sandbox $FNAME $DEF
    singularity build --fakeroot --sandbox $FNAME $DEF
}

## EOF
