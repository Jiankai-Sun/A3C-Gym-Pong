## A3C-Gym-Pong

## Training and Evaluation
The parameters for training and evaluation are defined in ```constants.py```. The most important parameter is ```TASK_LIST```, which is a dictionary that defines the scenes and targets to be trained and evaluated on. The keys of the dictionary are scene names, and the values are a list of location ids in the scene dumps, i.e., navigation targets. We use a type of asynchronous advantage actor-critic model, similar to [A3C](https://arxiv.org/abs/1602.01783), where each thread trains for one target of one scene. Therefore, make sure the number of training threads ```PARALLEL_SIZE``` is *at least* the same as the total number of targets. You can use more threads to further parallelize training. For instance, when using 8 threads to train 4 targets, 2 threads will be allocated to train each target.

The model checkpoints are stored to ```CHECKPOINT_DIR```, and Tensorboard logs are written in ```LOG_FILE```. To train a target-driven navigation model, run the following script:
```bash
# train a model for targets defined in TASK_LIST
python train.py
```

For evaluation, we run 100 episodes for each target and report the mean/stddev length of the navigation trajectories. To evaluate a model checkpoint in ```CHECKPOINT_DIR```, run the following script:
```bash
# evaluate a checkpoint on targets defined in TASK_LIST
python evaluate.py
```

## Acknowledgements
I would like to acknowledge the following references that have offered great help for me to implement the model.
* ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [David Silver's Deep RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [muupan's async-rl repo](https://github.com/muupan/async-rl/wiki)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)

