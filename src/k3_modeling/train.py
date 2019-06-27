from keras.optimizers import Adam
from src.k3_modeling.build_model import model
from src.k2_data_preparation.data_preparation import train_generator

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10)