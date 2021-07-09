from collections import deque
import copy

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Observation
from geese.constants import ACTIONLIST, NUM_GEESE
from geese.structure.train_data import TrainData
from geese.util.converter import action2int
from typing import Any, Deque, List, Tuple

from geese.structure.parameter.ppo_parameter import PPOParameter
import numpy as np


def calc_gae_list(delta_q: List[Deque], gae_param: np.ndarray) -> List[float]:
    """delta_qにある各DequeからそれぞれのGAEを算出する

    Args:
        delta_q (List[Deque]): deltaをqueに保存したlist.Deque内にあるdeltaは時系列順に保存されている
        gae_param (np.ndarray): GAE算出のために使用するパラメータのndarray.(1, λγ, (λγ)^2,・・・,(λγ)^(n-1))

    Returns:
        List[float]: 算出した各GAEのリスト
    """
    return [calc_gae(d_q, gae_param) for d_q in delta_q]


def calc_gae(d_q: Deque, gae_param: np.ndarray) -> float:
    """キューに保存されている値からGAEを算出する

    Args:
        d_q (Deque): deltaが保存されているDeque
        gae_param (np.ndarray): GAE算出のために使用するパラメータのndarray.(1, λγ, (λγ)^2,・・・,(λγ)^(n-1))

    Returns:
        float: 算出したGAE
    """
    return np.sum(np.array(d_q) * gae_param)


def add_delta_list(
    delta_que: List[Deque],
    reward_list: List[float],
    v_old_list: List[float],
    v_new_list: List[float],
    gamma: float,
) -> None:
    """[summary]

    Args:
        delta_que (List[Deque]): [description]
        reward_list (List[float]): [description]
        v_old_list (List[float]): [description]
        v_new_list (List[float]): [description]
        gamma (float): [description]
    """
    [
        add_delta(d_q, reward, v_new, v_old, gamma)
        for reward, v_old, v_new, d_q in zip(
            reward_list, v_old_list, v_new_list, delta_que
        )
    ]


def add_delta(
    d_q: Deque, reward: float, v_new: float, v_old: float, gamma: float
) -> None:
    """今回δを算出し、キューの右側に追加する

    Args:
        d_q (Deque): 追加する対象のDeque
        reward (float): 時刻tの行動により取得したリワード
        v_new (float): 時刻t+1のバリュー
        v_old (float): 時刻tのバリュー
        gamma (float): 割引率
    """
    d_q.append(reward + v_new * gamma - v_old)


def update_PPO_list(
    train_data: TrainData,
    obs: List[Observation],
    action: List[int],
    gae_list: List[np.ndarray],
    value: List[float],
    prob: List[np.ndarray],
    player_done_list: List[bool],
) -> None:
    """今回の行動で取得したデータをAIに学習させるためのデータとして追加する.
    ゲームが終了しているagentの情報は追加しない.

    Args:
        train_data (TrainData): トレーニング用データが格納しているオブジェクト
        obs (List[Observation]): 今回の状態が格納されているリスト
        action (List[int]): 今回の行動が格納されているリスト
        gae_list (List[float]): 今回のGAEが格納されているリスト
        value (List[float]): 今回のバリューが格納されているリスト
        prob (List[np.ndarray]): 今回の確率ベクトルが格納されているリスト
        player_done_list (List[bool]): ゲームが終了しているかどうかが格納されているリスト
    """
    train_data.obs_list.extend([o for o, d in zip(obs, player_done_list) if not d])
    train_data.action_list.extend(
        [a for a, d in zip(action, player_done_list) if not d]
    )
    train_data.gae_list.extend(
        [gae for gae, d in zip(gae_list, player_done_list) if not d]
    )
    train_data.v_list.extend([v for v, d in zip(value, player_done_list) if not d])
    train_data.pi_list.extend([p for p, d in zip(prob, player_done_list) if not d])


def create_que_list(index_1: int, index_2: int) -> List[List[Deque]]:
    """空のDequeが格納されている二次元リストを作成する

    Args:
        index_1 (int): 一次元目のサイズ
        index_2 (int): 二次元目のサイズ

    Returns:
        List[List[Deque]]: 作成したリスト
    """
    return [[deque() for _ in range(index_2)] for __ in range(index_1)]


def reset_que(index: int) -> List[Deque]:
    """空のDequeが格納されているリストを作成する

    Args:
        index (int): リストのサイズ

    Returns:
        List[Deque]: 作成したリスト
    """
    return [deque() for _ in range(index)]


def reset_train_data(train_data: TrainData) -> None:
    """学習用に格納しているデータを空にする

    Args:
        train_data (TrainData): 学習用データが格納されているオブジェクト
    """
    train_data.obs_list = []
    train_data.action_list = []
    train_data.gae_list = []
    train_data.v_list = []
    train_data.pi_list = []


def add_to_que_list(
    traget_que_list: List[List[Deque]], add_data_list: List[List[Any]]
) -> None:
    """traget_que_list内の各Dequeに今回のadd_dataを追加する.
    traget_que_list内のi,j要素のDequeには、add_data_listのi,j要素のデータが追加される

    Args:
        traget_que_list (List[List[Deque]]): 被追加対象となるDequeが格納されているリスト
        add_data_list (List[List[Any]]): 追加対象となる要素が格納されているリスト
    """
    [
        add_to_que(target_q, add_data)
        for target_q, add_data in zip(traget_que_list, add_data_list)
    ]


def add_to_que(target_q: List[Deque], add_data: List[Any]) -> None:
    """target_q内の各Dequeに対応するadd_dataの各要素を追加する

    Args:
        target_q (List[Deque]): 被追加対象となるDequeが格納されているリスト
        add_data (List[Any]): 追加対象となる要素が格納されているリスト
    """
    [t_q.append(a_d) for t_q, a_d in zip(target_q, add_data)]


def create_padding_data(
    ppo_parameter: PPOParameter,
    train_data: TrainData,
    obs_q: Deque,
    action_q: Deque,
    reward_q: Deque,
    delta_q: Deque,
    value_q: Deque,
    prob_q: Deque,
) -> None:
    """終了したゲームのデータに対して、n回分パディングしてトレーニングデータを作成する

    Args:
        ppo_parameter (PPOParameter): パラメータが格納されているオブジェクト
        train_data (TrainData): トレーニングデータが格納されているオブジェクト
        obs_q (Deque): 時系列順にゲームの各状態が格納されているキュー
        action_q (Deque): 時系列順にとった行動が格納されているキュー
        reward_q (Deque): 時系列順に取得したリワードが格納されているキュー
        delta_q (Deque): 時系列順に算出されたδが格納されているキュー
        value_q (Deque): 時系列順に算出されたバリューが格納されているキュー
        prob_q (Deque): 時系列順に算出された確率ベクトルが格納されているキュー

    Raises:
        ValueError: 渡した各キューの長さが異なっている場合、エラーとする
    """

    if (
        len(obs_q) != len(action_q)
        or len(obs_q) != len(value_q)
        or len(obs_q) != len(prob_q)
    ):
        raise ValueError

    add_delta(delta_q, reward_q[-1], value_q[-1], 0.0, ppo_parameter.gamma)
    if len(delta_q) != ppo_parameter.num_step:
        target_delta_q = copy.deepcopy(delta_q)
        while len(target_delta_q) != ppo_parameter.num_step:
            target_delta_q.append(0.0)
    else:
        target_delta_q = delta_q

    for _ in range(len(obs_q)):
        obs = obs_q.popleft()
        action = action2int(action_q.popleft())
        gae = calc_gae(target_delta_q, ppo_parameter.gamma)
        value = value_q.popleft()
        prob = prob_q.popleft()
        update_PPO_list(train_data, [obs], [action], [gae], [value], [prob], [False])

    # ダミーデータの投入
    obs_q.append(obs)
    action_q.append(ACTIONLIST[0])
    value_q.append(value)
    prob_q.append(prob_q)

    # ゴミ捨て
    delta_q.popleft()


def reshape_step_list(
    action_list: List[Action], value_n_list: np.ndarray, prob_list: np.ndarray
) -> Tuple[List[List[Action]], List[np.ndarray], List[np.ndarray]]:
    """受け取ったリストを二次元リストに変換する.
    一次元目には、異なるゲームが、二次元目には、同一ゲームの各Agentのデータが格納される

    Args:
        action_list (List[Action]): 各アクションが格納されているリスト
        value_n_list (np.ndarray): 各バリューが格納されているリスト
        prob_list (np.ndarray): 各確率ベクトルが格納されているリスト

    Returns:
        Tuple[List[List[Action]], List[np.ndarray], List[np.ndarray]]: [description]
    """
    reshape_action_list = [
        action_list[i : i + NUM_GEESE] for i in range(0, len(action_list), NUM_GEESE)
    ]
    reshape_value_n_list = [
        value_n_list[i : i + NUM_GEESE] for i in range(0, len(value_n_list), NUM_GEESE)
    ]
    reshape_prob_list = [
        prob_list[i : i + NUM_GEESE] for i in range(0, len(prob_list), NUM_GEESE)
    ]

    return reshape_action_list, reshape_value_n_list, reshape_prob_list


def update_self_PPO_list(
    r_q: Deque,
    o_q: Deque,
    a_q: Deque,
    v_q: Deque,
    p_q: Deque,
    d_q: Deque,
    gae_param: np.ndarray,
    train_data: TrainData,
) -> None:
    """

    Args:
        r_q (Deque): 時系列順に取得したリワードが格納されているキュー
        o_q (Deque): 時系列順にゲームの状態が格納されているキュー
        a_q (Deque): 時系列順に選択した行動が格納されているキュー
        v_q (Deque): 時系列順に算出されたバリューが格納されているキュー
        p_q (Deque): 時系列順に算出された確率ベクトルが格納されているキュー
        d_q (Deque): 時系列順にゲームの終了状態が格納されているキュー
        gae_param (np.ndarray): GAE算出に使用するパラメータ
        train_data (TrainData): トレーニングデータが格納されているオブジェクト
        before_done (bool): 一つ前の時間軸での終了状態
    """
    r_q.popleft()

    o = o_q.popleft()
    a = action2int(a_q.popleft())
    v = v_q.popleft()
    p = p_q.popleft()
    gae = calc_gae(d_q, gae_param)
    d_q.popleft()

    update_PPO_list(train_data, [o], [a], [gae], [v], [p], [False])
