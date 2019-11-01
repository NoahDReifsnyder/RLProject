import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses
import keras
model = Sequential([
    Dense(32, input_shape=(84, )),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.compile(loss=losses.mean_absolute_error, optimizer=sgd, metrics=['accuracy'])
env = gym.make('StarGunner-v0')
env = gym.wrappers.AtariPreprocessing(env)
env.reset()
space = env.action_space

obs = []
done = False
while not done:
    env.render()
    obv, reward, done, info = env.step(env.action_space.sample())  # take a random action
    obs.append((obv, reward))

env.close()

