from pydantic import BaseModel


class DsTrainState(BaseModel):
    pass


class TrainState(BaseModel):
    last_epoch: int
    val_loss_min: float

