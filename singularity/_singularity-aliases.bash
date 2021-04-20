# $ echo $SINGULARITY_SIF_ROOT_PATH 
# /mnt/nvme/Semisupervised_Ripples/singularity/singularity/semi_ripples_05.def
# $ echo $SINGULARITY_SIF_PATH
# /mnt/nvme/Semisupervised_Ripples/singularity/singularity/semi_ripples_05.def.sif
# $ echo $SINGULARITY_SIF_ROOT_NAME
# singularity/semi_ripples_05.def

# export SINGULARITY_SIF_ROOT=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
# export SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif

export SINGULARITY_SIF_PATH=$PJ_HOME/singularity/${SINGULARITY_FNAME}.sif

################################################################################
## Singularity image file version
################################################################################
# singularity shell
function sshell() {
    echo singularity shell --nv $SINGULARITY_SIF
    singularity shell --nv $SINGULARITY_SIF
}

# singularity python
function spy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH python $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH python $@
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
    DEF_FPATH=$1
    OPTION=$2
    
	SINGULARITY_SIF_ROOT_NAME=$DEF_FPATH # semi_ripples
	SINGULARITY_SIF_ROOT=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
	SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif
    
    echo singularity build $OPTION $SINGULARITY_SIF_PATH $SINGULARITY_SIF_ROOT.def
    singularity build $OPTION $SINGULARITY_SIF_PATH $SINGULARITY_SIF_ROOT.def
}

################################################################################
## Sandbox version
################################################################################
# singularity shell
function sshellw() {
    echo singularity shell --nv $SINGULARITY_SIF_ROOT
    singularity shell --nv $SINGULARITY_SIF_ROOT
}
# alias sshellw="singularity shell --writable $SINGULARITY_SIF_ROOT"


function sbuildw () {
    DEF_FPATH=$1
    # SEMI_RIPPLE_HOME=/mnt/nvme/Semisupervised_Ripple
	SINGULARITY_SIF_ROOT_NAME=$DEF_FPATH # semi_ripples
	SINGULARITY_SIF_ROOT=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
	SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif
    
    echo singularity build --fakeroot --sandbox \
         ${SINGULARITY_SIF_ROOT} ${SINGULARITY_SIF_ROOT}.def
    singularity build --fakeroot --sandbox \
         ${SINGULARITY_SIF_ROOT} ${SINGULARITY_SIF_ROOT}.def
}


function spyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT python $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT python $@
}


function sipyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT ipython $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT ipython $@
}

function sjpyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT jupyter-notebook $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT jupyter-notebook $@
}


## EOF
