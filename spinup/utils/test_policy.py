import time
import joblib
import os
import os.path as osp
import torch
from spinup import EpochLogger


def load_policy_and_env(fpath, itr='last', deterministic=False, render=False):
    """
    Load a PyTorch policy from save, along with RL env.
    """

    # handle which epoch to load from
    if itr=='last':
        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        import gymnasium as gym
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
        # gymnasium requires render_mode to be set at creation time;
        # recreate the env with 'human' render mode if rendering is requested.
        if render and env is not None:
            # spec.id may be None after unpickling; recover from registry
            if env.spec is not None:
                env_id = env.spec.id
            else:
                cls_name = type(env.unwrapped).__name__
                matches = sorted([k for k, v in gym.envs.registry.items()
                                   if cls_name in str(v.entry_point)])
                assert matches, f"Could not find a registered env matching {cls_name}"
                # Filter by continuous/discrete to pick the right variant
                is_continuous = getattr(env.unwrapped, 'continuous', False)
                env_id = None
                for match in reversed(matches):  # Start from highest version
                    # Check if this match is the right type (continuous vs discrete)
                    # by looking at the environment ID pattern
                    if is_continuous:
                        # For continuous, prefer envs with "Continuous" in the name
                        if 'Continuous' in match:
                            env_id = match
                            break
                    else:
                        # For discrete, prefer envs WITHOUT "Continuous" in the name
                        if 'Continuous' not in match:
                            env_id = match
                            break
                # Fallback to last match if no perfect match found
                if env_id is None:
                    env_id = matches[-1]
            env = gym.make(env_id, render_mode='human')
    except Exception as e:
        print(f"Warning: could not load environment ({e})")
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname, weights_only=False)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, _ = env.reset()
    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, _ = env.reset()
            r, d, ep_ret, ep_len = 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic,
                                          render=not(args.norender))
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))