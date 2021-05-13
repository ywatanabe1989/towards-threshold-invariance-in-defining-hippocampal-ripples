################################################################################
## Singularity image file version
################################################################################
# export SINGULARITY_SIF_ROOT=$PJ_HOME/singularity/${SINGULARITY_FNAME}
# export SINGULARITY_SIF_PATH=$PJ_HOME/singularity/${SINGULARITY_FNAME}.sif

export SINGULARITY_SIF_ROOT=$PJ_HOME/singularity/${SINGULARITY_FNAME}
export SINGULARITY_SIF_PATH=$PJ_HOME/singularity/${SINGULARITY_FNAME}.sif

function gt_sif_root() {
    PWD=`pwd`

    if [[ "$PWD" == *"proj"* ]]; then
        PJ_DIR="${PWD%proj/*}"proj/

        CURRENT_PJ_NAME="${PWD##*proj/}"
        CURRENT_PJ_NAME="${CURRENT_PJ_NAME%/*}"

        CURRENT_PJ=${PJ_DIR}$CURRENT_PJ_NAME

        SINGULARITY_SIF_ROOT=$CURRENT_PJ/singularity/image
        echo $SINGULARITY_SIF_ROOT

    else
        echo $SINGULARITY_SIF_ROOT

    fi
}


################################################################################
## Singularity image file version
################################################################################
# singularity shell
function sshell() {
    echo singularity shell --nv `gt_sif_root`.sif
    singularity shell --nv `gt_sif_root`.sif
}

# singularity python
function spy () {
    echo "singularity exec --nv `gt_sif_root`.sif python3 $@" &&
    singularity exec --nv `gt_sif_root`.sif python3 $@
}

# singularity ipython
function sipy () {
    echo "singularity exec --nv `gt_sif_root`.sif ipython $@" &&
    singularity exec --nv `gt_sif_root`.sif ipython $@
}

# singularity jupyter
function sjpy () {
    echo "singularity exec --nv `gt_sif_root`.sif jupyter-notebook $@" &&
    singularity exec --nv `gt_sif_root`.sif jupyter-notebook $@
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
    
    echo singularity shell --nv `gt_sif_root`
    singularity shell --nv `gt_sif_root`
}

# writable singularity python
function spyw () {
    echo "singularity exec --nv `gt_sif_root` python3 $@" &&
    singularity exec --nv `gt_sif_root` python3 $@
}

# writable singularity ipython
function sipyw () {
    echo "singularity exec --nv `gt_sif_root` ipython $@" &&
    singularity exec --nv `gt_sif_root` ipython $@
}

# writable singularity jupyter-notebook
function sjpyw () {
    echo "singularity exec --nv `gt_sif_root` jupyter-notebook $@" &&
    singularity exec --nv `gt_sif_root` jupyter-notebook $@
}

# writable singularity build
function sbuildw () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --fakeroot --sandbox $FNAME $DEF
    singularity build --fakeroot --sandbox $FNAME $DEF
}

function sbuildwr () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --remote --sandbox $FNAME $DEF
    singularity build --remote --sandbox $FNAME $DEF
}

## EOF
