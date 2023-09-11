# edit in 2023/09/07
# register for new environment with more distractors
from babyai.levels.iclr19_levels import Level_GoTo
import gym
# register for new environment with more distractors

class Level_GoToS8R2D1(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=8, num_rows=2, num_cols=2, seed=seed)

class Level_GoToS8R2D2(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=2, room_size=8, num_rows=2, num_cols=2, seed=seed)

class Level_GoToS8R2D4(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=4, room_size=8, num_rows=2, num_cols=2, seed=seed)

class Level_GoToS8R2D8(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=8, room_size=8, num_rows=2, num_cols=2, seed=seed)

class Level_GoToS8R2D16(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=16, room_size=8, num_rows=2, num_cols=2, seed=seed)

class Level_GoToS8R2D32(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=32, room_size=8, num_rows=2, num_cols=2, seed=seed)



def register_levels(module_name, globals):
    """
    Register OpenAI gym environments for all levels in a file
    """

    # Iterate through global names
    for global_name in sorted(list(globals.keys())):
        if not global_name.startswith('Level_'):
            continue

        level_name = global_name.split('Level_')[-1]
        level_class = globals[global_name]

        # Register the levels with OpenAI Gym
        gym_id = 'MyBabyAI-%s-v0' % (level_name)
        entry_point = '%s:%s' % (module_name, global_name)
        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        # Add the level to the dictionary
        # level_dict[level_name] = level_class

        # Store the name and gym id on the level class
        level_class.level_name = level_name
        level_class.gym_id = gym_id

# Register the levels in this file
register_levels(__name__, globals())