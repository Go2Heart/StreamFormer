name="streamformer_multitask_with_flow"

python tools/train_net.py \
    --config configs/THUMOS/MAT/${name}.yaml \
    SOLVER.PHASES "['train', 'test']"
