if [ -z "$DATA" ]
then
    export DATA=aokvqa
fi

if [ -z "$SPLIT" ]
then
    export SPLIT=val
fi

if [ -z "$NUM_EXAMPLES"]
then
    export NUM_EXAMPLES=-1
fi

if [ -z "$TASK_TYPE" ]
then
    export TASK_TYPE=direct_answer
fi

echo "DATA: $DATA"
echo "SPLIT: $SPLIT"
echo "NUM_EXAMPLES: $NUM_EXAMPLES"
echo "TASK_TYPE: $TASK_TYPE"

python -m recoverr_011924 \
    --config_file ${1} \
    --dataset aokvqa \
    --split val \
    --task_type direct_answer