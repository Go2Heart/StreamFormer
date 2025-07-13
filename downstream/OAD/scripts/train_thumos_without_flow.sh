name="streamformer_multitask_without_flow"

python tools/train_net.py \
    --config configs/THUMOS/MAT/${name}.yaml \
    SOLVER.PHASES "['train', 'test']"
