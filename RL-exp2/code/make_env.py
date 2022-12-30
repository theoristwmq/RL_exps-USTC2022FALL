import time

import gym
import yaml


from wrapper import *



wrapper_dict = {
    "WarpReward": WarpReward,
    "WarpDone": WarpDone,
    "TimeLimitWrapper": TimeLimitWrapper,
    "NeverStopWrapper": NeverStopWrapper,
    "WarpFrameWrapper": WarpFrameWrapper,
    "FrameStackWrapper": FrameStackWrapper,
    "WrapAction": WrapAction,
    "ExpandWrapper": ExpandWrapper,
    'DisplayWrapper': DisplayWrapper,
    'InfoExpandWrapper': InfoExpandWrapper,
}
def read_yaml(file: str) -> dict:
    try:
        file = open(file, 'r', encoding="utf-8")
        # 读取文件中的所有数据
        file_data = file.read()
        file.close()
        # 指定Loader
        dict_original = yaml.load(file_data, Loader=yaml.FullLoader)

        return dict_original
    except:
        return {}


def make_gymenv(file):
    cfg = read_yaml(file)
    env = gym.make(cfg['env_name'])

    for wrapper in cfg['wrapper']:
        env = wrapper_dict[wrapper](env, cfg)
    return env


if __name__ == "__main__":
    env = make_gymenv('env.yaml')
    s = env.reset()
    print(s.shape)
    print(env.action_space)
    done = False
    while not done:
        next_state, reward, done, _ = env.step(env.action_space.sample())
        print(reward, done)
        env.render()
        time.sleep(1)
