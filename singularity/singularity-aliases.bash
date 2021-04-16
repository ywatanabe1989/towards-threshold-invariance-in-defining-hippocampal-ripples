export SINGULARITY_SIF_ROOT_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
export SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif


## Using *.sif
function sshell() {
    echo singularity shell --nv $SINGULARITY_SIF_ROOT_PATH
    singularity shell --nv $SINGULARITY_SIF_ROOT_PATH
}


function spy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH python $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH python $@
}


function sipy () {
    echo "singularity exec --nv $SINGULARITY_SIF_PATH ipython $@" &&
    singularity exec --nv $SINGULARITY_SIF_PATH ipython $@
}


function sbuild () {
    DEF_FPATH=$1
    OPTION=$2
    
    # SEMI_RIPPLE_HOME=/mnt/nvme/Semisupervised_Ripples
	SINGULARITY_SIF_ROOT_NAME=$DEF_FPATH # semi_ripples
	SINGULARITY_SIF_ROOT_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
	SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif
    
    echo singularity build $OPTION SINGULARITY_SIF_PATH SINGULARITY_SIF_ROOT_PATH.def
    singularity build $OPTION SINGULARITY_SIF_PATH SINGULARITY_SIF_ROOT_PATH.def
}


## Using sandbox
alias sshellw="singularity shell --writable $SINGULARITY_SIF_ROOT_PATH"


function sbuildw () {
    DEF_FPATH=$1
    # SEMI_RIPPLE_HOME=/mnt/nvme/Semisupervised_Ripple
	SINGULARITY_SIF_ROOT_NAME=$DEF_FPATH # semi_ripples
	SINGULARITY_SIF_ROOT_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}
	SINGULARITY_SIF_PATH=$SEMI_RIPPLE_HOME/singularity/${SINGULARITY_SIF_ROOT_NAME}.sif
    
    echo singularity build --fakeroot --sandbox \
         ${SINGULARITY_SIF_ROOT_PATH} ${SINGULARITY_SIF_ROOT_PATH}.def
    singularity build --fakeroot --sandbox \
         ${SINGULARITY_SIF_ROOT_PATH} ${SINGULARITY_SIF_ROOT_PATH}.def
}


function spyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT_PATH python $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT_PATH python $@
}


function sipyw () {
    echo "singularity exec --nv $SINGULARITY_SIF_ROOT_PATH ipython $@" &&
    singularity exec --nv $SINGULARITY_SIF_ROOT_PATH ipython $@
}


## EOF
