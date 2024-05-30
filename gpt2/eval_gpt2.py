from imports import *

def eval_on_vanilla_gpt(DEVICE, model, data):
    
    model.eval()
    model.to(DEVICE)
    
    with model.trace() as tracer:
        pass