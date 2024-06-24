from torch.optim import RAdam, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup

# Choose optimizer and learning rate scheduler
def getOptimizer(model, total_train_steps, scheduler_type="linear",
                  lr=1e-4, weight_decay=0.01, warmup_steps=0):
    
    optimizer = AdamW(params=model.parameters(), lr=float(lr), weight_decay=weight_decay)
    
    if scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps
        )
    elif scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )

    return optimizer, lr_scheduler