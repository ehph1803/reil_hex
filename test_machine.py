

def machine(board, action_space):
    from joblib import load
    import torch
    
    actor = load('v1_hex_actor.a2c')
    probs = actor.forward([1,2,3,4,5,6])
    #print(action_space)
    played_logic = probs.detach().clone()

    for (x, y) in action_space:
        played_logic[x*7 + y] = 0

    #print(played_logic)

    new_probs = torch.sub(probs, played_logic)

    #print(new_probs)

    dist = torch.distributions.Categorical(probs=new_probs)
    action = dist.sample()
    
    x = int(action.detach().numpy() / 7)
    y = int(action.detach().numpy() % 7)
    action = (x, y)

    return action
