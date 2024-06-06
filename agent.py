import torch
import random
import numpy as np
from snake_game import SnakeGameAI, Direction, Point
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def _init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None


    def get_state(self, snake_game):
        head = snake_game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake_game.direction == Direction.LEFT
        dir_r = snake_game.direction == Direction.RIGHT
        dir_u = snake_game.direction == Direction.UP
        dir_d = snake_game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snake_game.is_collision(point_r)) or
            (dir_l and snake_game.is_collision(point_l)) or
            (dir_u and snake_game.is_collision(point_u)) or
            (dir_d and snake_game.is_collision(point_d)) ,

            # Danger right
            (dir_u and snake_game.is_collision(point_r)) or
            (dir_d and snake_game.is_collision(point_l)) or
            (dir_l and snake_game.is_collision(point_u)) or
            (dir_r and snake_game.is_collision(point_d)) ,

            # Danger left
            (dir_d and snake_game.is_collision(point_r)) or
            (dir_u and snake_game.is_collision(point_l)) or
            (dir_r and snake_game.is_collision(point_u)) or
            (dir_l and snake_game.is_collision(point_d)) ,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            snake_game.food.x < snake_game.head.x,
            snake_game.food.x > snake_game.head.x,
            snake_game.food.y < snake_game.head.y,
            snake_game.food.y > snake_game.head.y
        ]

        return np.array(state, dtype=int)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)   
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        #for state, action, reward, next_state, done in mini_sample:
            #self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        pass

    def train():
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        snake_game = SnakeGameAI()
        while True:
            state_old = agent.get_state(snake_game)

            final_move = agent.get_action(state_old)

            reward, done, score = snake_game.play_step(final_move)

            state_new =  agent.get_state(snake_game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                snake_game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    
                print('Game', agent.n_games, 'Score', score, 'Record:', record)





    if __name__=='__main__':
        train()
