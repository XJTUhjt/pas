from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import numpy as np
from .Queue import Queue



class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section, policy=None):
        super().__init__(config, section, policy)
        if policy == 'none':
            self.isObstacle = True
        else:
            self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent

        #每个human都要保存自己的历史状态,len可作为参数传入
        self.history_visible_states = Queue(maxsize=config.humans.max_saved_states_length)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        if self.isObstacle:
            action =  ActionXY(0., 0.)
        else:
            action = self.policy.predict(state)
        return action
    
    #保存当前可见状态进入队列,队列元素内容为ObservableState与时间戳
    def save_visible_state(self,human_id, timestamp):
        self.history_visible_states.enqueue(np.array([human_id, self.px, self.py, self.vx, self.vy, self.theta, self.radius, timestamp]))

    #经过几个时间戳不可见则清空历史状态队列
    def clear_self_history_states(self):
        self.history_visible_states.clear()

    def get_history_states_np(self):
        if not self.history_visible_states.is_empty():
            #查看队列的数据再塞回去
            elements_in_queque = []
            while not self.history_visible_states.is_empty():
                element = self.history_visible_states.dequeue()
                elements_in_queque.append(element)
            for element in elements_in_queque:
                self.history_visible_states.enqueue(element)
            
            #堆叠历史状态，t时刻在最上面
            result_history_matrix = np.vstack(elements_in_queque[::-1])

            #填充放到rotate之后
            # #maxsize-现在有的states拼接成 feature_dims * maxsize 维度的二维矩阵
            # for lack_num in range(8 - elements_matrix.shape[0]):
            #     result_history_matrix = np.vstack(elements_matrix, np.array([elements_matrix[0][0], 0, 0, 0, 0, 0, 0, -1]))

        else:
            lack_id_col = np.full((8,1), 999)
            lack_data_matrix = np.full((8, 6), 0)
            lack_timestamp = np.full((8,1), -1)

            result_history_matrix = np.concatenate((lack_id_col, lack_data_matrix, lack_timestamp), axis=1)
                

        return result_history_matrix