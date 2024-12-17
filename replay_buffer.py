from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state,
            action,
            reward,
            next_state,
            done
        )
        self.position = (self.position + 1) % self.capacity
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]