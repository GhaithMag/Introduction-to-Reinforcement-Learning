import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

class GameEnv:
    def __init__(self):
        self.board_size = 10
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        #Obstacle de niveau 1, mur
        self.obstacles_1 = [(4, 8), (4, 2), (7, 3), (3, 5), (8, 0)]
        self.obstacles_2 = [(6, 6), (3, 4), (4, 4), (7, 7)]
        self.obstacles_3 = [(4, 9), (3, 2)]

        self.numbre_de_tour = 0

        self.max_steps = 300
        
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.p1_pos = (0, 0)
        
        self.board[self.p1_pos] = 1
        for obs in self.obstacles_1:
            self.board[obs] = -1

        for obs in self.obstacles_2:
            self.board[obs] = -50

        for obs in self.obstacles_3:
            self.board[obs] = 20
        return self.get_state()

    def get_state(self):
        state = np.zeros((4, self.board_size, self.board_size), dtype=int)
        state[0][self.p1_pos] = 1

        for obs in self.obstacles_1:
            y, x = obs
            state[1][y][x] = 1

        for obs in self.obstacles_2:
            y, x = obs
            state[2][y][x] = 1

        for obs in self.obstacles_3:
            y, x = obs
            state[3][y][x] = 1

        return state

    def reset_turns(self):
        self.numbre_de_tour = 0

    def step(self, player, action):
        new_pos = self.move(self.p1_pos, action)

        if new_pos in self.obstacles_1:
            self.p1_pos = self.p1_pos
            mur = 'Positif'
        else:
            self.p1_pos = new_pos
            mur = 'RAF'

        self.numbre_de_tour += 1  # Augmenter le nombre d'actions effectuées
        reward, done = self.get_reward(player, mur)
        return self.get_state(), reward, done



    def move(self, pos, action):
        y, x = pos
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(self.board_size - 1, y + 1)
        elif action == 2:  # Left
            x = max(0, x - 1)
        elif action == 3:  # Right
            x = min(self.board_size - 1, x + 1)
        new_pos = (y, x)
        return new_pos

    
    def render(self):
        print("\n")
        for i in range(self.board_size):
            row = ""
            for j in range(self.board_size):
                if (i, j) == self.p1_pos:
                    row += "P1 "

                elif (i, j) in self.obstacles_1:
                    row += "O "
                elif (i, j) in self.obstacles_2:
                    row += "X "

                elif (i, j) in self.obstacles_3:
                    row += "A "
                else:
                    row += ". "
            print(row)

    def get_reward(self, player, mur):
        reward, done = 0, False

        if self.p1_pos == (9, 9):
            reward, done = 100, True
        elif mur == 'Positif':
            reward = -1
        elif self.p1_pos in self.obstacles_2:
            self.p1_pos = (0, 0)
            reward, done = -50, True
        elif self.p1_pos in self.obstacles_3:
            self.p1_pos = (8, 9)
            reward = 10
        else:
            reward = 0.0

        if self.numbre_de_tour >= self.max_steps:  # Vérifier si le nombre d'actions dépasse le maximum
            reward = -100  # Attribuer une récompense négative
            done = True  # Terminer la partie

        return reward, done


    def play(self):
        self.reset()
        total_reward = 0
        done = False
        self.render()
        
        while not done:
            action = input("Entrez l'action (0 = haut, 1 = bas, 2 = gauche, 3 = droite): ")
            try:
                action = int(action)
                if action not in [0, 1, 2, 3]:
                    raise ValueError
            except ValueError:
                print("Entrez un nombre entier valide entre 0 et 3.")
                continue

            state, reward, done = self.step(1, action)
            total_reward += reward
            self.render()
            print(f"Récompense pour cette étape: {reward}")
            print(f"Récompenses cumulées: {total_reward}")

        print("Le jeu est terminé.")


"""if __name__ == "__main__":
    env = GameEnv()
    env.play()"""




class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 10 * 10)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class Agent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.model = DQN()
        self.target_model = DQN()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 4
        self.gamma = 0.98
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.save_path = f"model_player_{player_id}.pth"

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 3)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            target = self.model(state_tensor)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(self.target_model(next_state_tensor))

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        if file_name is None:
            file_name = self.save_path
        torch.save(self.model.state_dict(), file_name)

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
        self.update_target_model()

def train_agent(epochs=1000):
    env = GameEnv()
    agent = Agent(player_id=1)
    reward_history = []

    for e in range(epochs):
        total_reward = 0
        state = env.reset()
        env.reset_turns()
        print(f"Starting epoch {e+1}/{epochs}")
        done = False
        step_counter = 0 # Compteur d'étapes pour chaque épisode
        rec =[]
        while not done:
            
            step_counter += 1
            

            
            action = agent.act(state)
            next_state, reward, done = env.step(player=1, action=action)
            
            agent.memorize(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            rec.append(reward)
            
            if step_counter % 100 == 0:
                
                print(f"Epoch {e+1}/{epochs}, Step {step_counter}") # Ajouter un message pour indiquer une nouvelle étape
                print("Raward totale:", total_reward)
                

            agent.replay()

        reward_history.append(total_reward)

        mean_reward = np.mean(reward_history[-100:])
            
        print(f"Epoch {e+1}/{epochs} - Total reward for player : {total_reward}")
        print(f"Epoch {e+1}/{epochs} - Mean reward for player : {mean_reward}")

       
        

        if e % 5 == 0:
            agent.update_target_model()
            
        if e % 10 == 0:
            torch.save(agent.model.state_dict(), "model_custom_name.pth")
            
    # À la fin de la fonction train_agent()
    torch.save(agent.model.state_dict(), "model_custom_name.pth")


def test(env, agent):
    total_reward = 0
    state = env.reset()
    done = False
    step_counter = 0
    while not done:
        step_counter += 1
        action = agent.act(state)
        next_state, reward, done = env.step(player=1, action=action)
        total_reward += reward
        state = next_state
    print(f"Total reward during test: {total_reward}")
    print(f"Steps taken during test: {step_counter}")
    return total_reward, step_counter


train_agent()

if __name__ == "__main__":
    env = GameEnv()
    agent = Agent(player_id=1)
    agent.model.load_state_dict(torch.load("model_custom_name.pth"))
    test(env, agent)

