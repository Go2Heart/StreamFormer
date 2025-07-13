name="streamformer_multitask_without_flow"

python tools/train_net.py \
    --config configs/TVSeries/MAT/${name}.yaml \
    SOLVER.PHASES "['train', 'test']"
