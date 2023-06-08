def agent(board, action_space):
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, n_actions, device, in_channels=1, kernel_size=3):
            super().__init__()

            self.device = device

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten()
            )

            self.lin = nn.Sequential(
                nn.Linear(64, n_actions),
                nn.Softmax()
            )

        def forward(self, X):
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            # print(len(tensor))
            tensor.unsqueeze_(-1)
            tensor = tensor.expand(len(tensor), len(tensor), 1)
            tensor = tensor.permute(2, 0, 1)
            # print(tensor)

            x = self.conv(tensor)
            x = torch.transpose(x, 0, 1)
            x = self.lin(x)

            return x.flatten()

    def get_action(actor_, state, action_space_, recode_black_white=False):
        game = None

        if recode_black_white:
            from fhtw_hex.hex_engine import hexPosition
            game = hexPosition(len(board))
            action_space_ = [game.recode_coordinates(action) for action in action_space_]
            state = [list(row) for row in zip(*reversed(state))]
            state = [[j * -1 for j in i] for i in state]
        
        probs = actor_(state)

        # remove played spaces from probabilities
        free_logic = torch.zeros(probs.numel(), dtype=torch.bool)

        for (x, y) in action_space_:
            free_logic[x * len(state) + y] = 1

        new_probs = probs[free_logic]
        dist = torch.distributions.Categorical(probs=new_probs)
        a = dist.sample()
        a = action_space_[a.detach().numpy()]
        
        if recode_black_white:
            a = game.recode_coordinates(a)

        return a

    actor = Actor(len(board)**2, 'cpu')
    actor.load_state_dict(torch.load('fhtw_hex/submission/actor_final_state_dict.torch'))
    
    board_sum = 0
    
    
    for i in range(len(board)):
        board_sum += sum(board[i])
        
    #print(board_sum)
    
    action = action_space[0]
    
    if board_sum == 0:
        action = get_action(actor, board, action_space)
    else:
        action = get_action(actor, board, action_space, True)

    return action
