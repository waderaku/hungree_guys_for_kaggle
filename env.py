from kaggle_environments import make, evaluate


class Env():
    def __init__(
        num_rows=7,
        num_colums=11,
        action_list=[]
    ):


env = make("hungry_geese", debug=False)
env.reset()
trainer = env.train([None, "greedy", "greedy", "greedy"])

test = trainer.reset()

trainer.step(1)

pass
