################################################################################
## Switches singularity container according to the current directory
################################################################################
'''
Example:


    /home/<USERNAME>/proj/PROJECT_A/singularity/proj_A_20210918.sif
    /home/<USERNAME>/proj/PROJECT_A/singularity/image.sif (-> softlink to .proj_A_20210918.sif)

    /home/<USERNAME>/proj/PROJECT_B/singularity/proj_B_20210815.sif
    /home/<USERNAME>/proj/PROJECT_B/singularity/image.sif (-> softlink to .proj_A_20210815.sif)
    ...
    /home/<USERNAME>/proj/PROJECT_Z/singularity/image.sif

    $ cd /home/<USERNAME>/proj/PROJECT_B/
    $ sshell # shell into the PROJECT_B singularity container
'''

export SINGULARITY_SIF_ROOT=$PJ_HOME/singularity/image
export SINGULARITY_SIF_PATH=$PJ_HOME/singularity/image.sif

function get_sif_root_from_the_current_dir() {
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
## Singularity image file versions
################################################################################
# singularity shell
function sshell() {
    echo singularity shell --nv `get_sif_root_from_the_current_dir`.sif
    singularity shell --nv `get_sif_root_from_the_current_dir`.sif
}

# singularity python
function spy () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir`.sif python3 $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir`.sif python3 $@
}

# singularity ipython
function sipy () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir`.sif ipython $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir`.sif ipython $@
}

# singularity jupyter
function sjpy () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir`.sif jupyter-notebook $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir`.sif jupyter-notebook $@
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
## Sandbox versions
################################################################################
# writable singularity shell
function sshellw() {
    
    echo singularity shell --nv `get_sif_root_from_the_current_dir`
    singularity shell --nv `get_sif_root_from_the_current_dir`
}

# writable singularity python
function spyw () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir` python3 $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir` python3 $@
}

# writable singularity ipython
function sipyw () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir` ipython $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir` ipython $@
}

# writable singularity jupyter-notebook
function sjpyw () {
    echo "singularity exec --nv `get_sif_root_from_the_current_dir` jupyter-notebook $@" &&
    singularity exec --nv `get_sif_root_from_the_current_dir` jupyter-notebook $@
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
