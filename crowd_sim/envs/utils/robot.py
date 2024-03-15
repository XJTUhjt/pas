from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointState_noV
import numpy as np
from .Queue import Queue



class Robot(Agent):
    def __init__(self, config,section):
        super().__init__(config,section)

        #每个human都要保存自己的历史状态,len可作为参数传入
        self.history_visible_states = Queue(maxsize=config.robot.max_saved_states_length)
        self.maxsize = config.robot.max_saved_states_length

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_noV(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState_noV(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action
    
    #保存当前可见状态进入队列,队列元素内容为ObservableState与时间戳
    def save_robot_state(self, timestamp):
        self.history_visible_states.enqueue(np.array([100, self.px, self.py, self.gx, self.gy, self.vx, self.vy, self.theta, self.radius, timestamp]))

    def clear_self_history_states(self):
        self.history_visible_states.clear()

    def get_history_states_np(self):
        assert not self.history_visible_states.is_empty()
        #查看队列的数据再塞回去
        elements_in_queque = []
        while not self.history_visible_states.is_empty():
            element = self.history_visible_states.dequeue()
            elements_in_queque.append(element)
        for element in elements_in_queque:
            self.history_visible_states.enqueue(element)
        
        #堆叠历史状态，t时刻在最上面
        elements_matrix = np.vstack(elements_in_queque[::-1])

        #maxsize-现在有的states拼接成 feature_dims * maxsize 维度的二维矩阵，填充放到rotate之后
        # for lack_num in range(8 - elements_matrix.shape[0]):
        #     result_history_matrix = np.vstack(elements_matrix, np.array([100, 0, 0, 0, 0, 0, 0, -1]))

        return elements_matrix
