generate and train a new model:
>>python run_unet.py train myNewModel.pkl

train an existing model, the model has to be saved in the model folder:
>>python run_unet.py train myOldModel.pkl

train an existing model, with specific learning rate:
>>python run_unet.py train myOldModel.pkl 0.001

validate an existing model, the model has to be saved in the model folder:
>>python run_unet.py validate myGoodModel.pkl

on Windows its "python" on linux its "python3"