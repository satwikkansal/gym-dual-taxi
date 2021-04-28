## An informal introduction to Reinforcement Learning

## Introduction

In 2016, AplhaGo, a program developed for playing the game of Go, made headlines when it beat the world champion Go player in a five-game match. This was a great feat because the number of possible legal moves in Go are of the order of 2.1 × 10<sup>170</sup>.  To put this in context, this number is far, far greater than the number of atoms in the observable universe which are of the order of 10<sup>80</sup>. Such a high number of possiblities make it almost impossible to create a program that can play effectively using brute-force or somewhat optimized search algorithms. A part of the secret sauce of AlphaGO was the usage of Reinforcement Learning to improve its understanding of the game by playing against itself. Since then, the field of Reinfocement Learning has seen a lot of interest and much more efficient programs have been developed to play various games at a pro-human efficiency. Although you would find Reinforcement Learning discussed in the context of Games and Puzzles in most places (including today), the applications of Reinforcement Learning are much wider, which you'll understand soon by the end of this tutorial, whose objective is to give you a gentle introduction to the world of Reinforcement Learning.  

## What is Reinforcement Learning

Reinforcement learning is a paradigm of Machine Learning where learning happens through the feedback gained by an agent's interaction with its environment. This is also one of the key differentiator of Reinforcement Learning with other two paradigms of Machine learning (Supervised learning and Unsupervised learning). Supervised learning algorithms require fully labeled-training-data, and Unsupervised learning algorithms need no labels. Reinforcement learning algorithms on the other hand, utilise feedback from the environment they're operating in to get better at the tasks they're being trained to perform. 

It is almost inevitable to talk about Reinforcement Learning with clarity without using some technical terms like "agent", "action", "state", "reward", and "environment". So let's try to gain a high-level understanding of Reinforcement Learning and these terms through an analogy,

### Understanding Reinforcement learning through Birbing

Let's watch first few seconds of this video of a talking parrot,

https://www.youtube.com/watch?v=u7TiRqh7x8s

Pretty cool, isn't it?

And now think about how did someone manage to teach this parrot to reply with certain sounds on certain prompts. And if you carefully observed, part of the answer lies in the food the parrot is given after every cool response. The human asks a question, and the parrot tries to respond in many different ways, and if the parrot's response is the desired one, it is rewarded with food.

Now guess what, the next time the parrot is exposed to the same cue, it is likely to answer similarly in expectation of more food. This is how we "reinforce" certain behaviors through positive experiences. If I had to explain the above process in terms of Reinforcement learning concepts, it'd be something like,

"The agent learns to take desired for a give state in the environment", where,

- The "agent" is the parrot
- The "state" is questions or cues the parrot is exposed to
- The "actions" are the sounds it is uttering 
- The "reward" is the food he gets when he takes a desired action
- And the "environment" is the place where parrot is living in (or in other words, everything else than the parrot)

The reinforcement can happen through negative experiences too. For example, if a child touches a burning candle out of curiosity, (s)he is unlikely to repeat the same action again. So in this case instead of reward, the agent got a penalty, which would disincentivize the agent to repeat the same action in future again.

If you try to think about it, there are countless similar real world analogies. This suggests why Reinforcement Learning can be  useful for a wide variety of real world application and why it might be path to create General AI Agents (think of a program that can not just beat a human in the game of Go, but multiple games like Chess, GTA, etc). It might still take a lot of time to develop agents with general intelligence, but reading about programs like [MuZero](https://en.wikipedia.org/wiki/MuZero) (one of the many successors of Alpha Go) hints that Reinforcement learning might have a decent role to play in achieving that.

After reading the analogies, a few questions like below might have come into your mind,

- Real world example is fine, but how do I do this "reinforcement" in the computer world?
- What are these algorithms, and how do they work?

Let's start answering such questions as switch gears and dive into certain technicalities of Reinforcement learning

## Example problem statement: Self driving taxi 

Wouldn't it be cool if we can train an agent (i.e. create a computer program) to pick up from a location and drop them to their desired location. In the rest of the tutorial, we'll try to solve a simplified version of this problem through reinforcement learning.

Before we do anything else, let's specify the typical steps in a Reinforcement learning process,

https://storage.googleapis.com/lds-media/documents/Reinforcement-Learning-Animation.gif

1. Agent observes the environment. The observation is represented in digital form and also called "state".
2. The agent utilises the observation to decide how to act. The strategy agent uses to figure out the action to perform is also referred to as "policy".
3. The agent performs the action in the environment
4. The environment as a result of the action, may move to a new state (i.e. generate different observations) and may return a feedback to the agent in the form of rewards/penalties. 
5. The agent uses the rewards and penalties to refine its policy
6. The process can be repeated until agent finds an optimal policy

Now that we're clear about the process, we need to set up the environment. In most cases what this means is we need to figure out the following details,

### 1. The state space

Typically, a "state" will encodes the observable information that the agent can use to learn to act efficiently. For example, in case of self-driving-taxi, the state information could contain the following information,

- The current location of the taxi
- The current location of the passenger
- The destination

There can be multiple ways to represent such information, and how one ends up doing it depends mostly on the level of sophistication intended. So the state space is the set of all possible states an environment can be in. 

For example, if we consider our environment for the self-driving taxi to be a two dimensional 4x4 grid, there are 

- 16 possible locations for the taxi
- 16 possible locations for the passenger
- and 16 possible destination

Which means, our state space size becomes 16 x 16 x 16 = 4096 i.e. at any point of time the environmend must be in either of these 4096 states. 

### 2. The action space

An action space is the set of all possible actions an agent can take in the environment. Taking the same 2D grid-world example, the taxi agent may be allowed to take following actions,

- Move North
- Move South
- Move East
- Move West
- Pickup
- Drop-off

Again, there can be multiple ways to define the action space, and this is just one of them. The choice also depends on the level of complexity and algorithms you'd want to use later.

### 3. The rewards 

The rewards and penalties are critical for agent's learning. While deciding the reward structure we need to carefully think about the magnitude, direction (positive or negative), and frequency of the reward (every time step / based on certain milestone / etc). Taking the same gridworld environment example, some ideas for reward structure can be,

- The agent should receive a positive reward when it performs a successful passenger dropoff. The reward should be high in magnitude because this behavior is highly desired.
- The agent should be penalized if it tries to drop off a passenger in wrong locations.
- The agent should get a slight negative reward for not making it to the destination after every time-step. This would incentivize the agent to take faster routes.

There can be more ideas for rewards like giving a reward for successful pickup and so on. 

### 4. The transition rules

The transition rules are kind of the brain of the environment which combine all the above three pieces together. They are often represented in terms of tables (a.k.a state transition tables) which specify that,

> For a given state S, if you take an action A, the new state of the environment becomes S' and the reward received is R. 

`#TODO:` Insert table here

An example row could be, when the taxi's location is bottom-right corner, and passenger's location is also bottom-right corner, and the agent takes the "pickup" action, give him a positive reward.

In real world the state transitions may not be deterministic i.e. they can be either

- Stochastic; which means the rules operate by probability i.e. if you take an action there's an X1% chance you'll end up in state S1, and Xn% chance you'd end up in a state Sn.
- Unknown; which means it is not known in advance what all possible states the agent can get into if it takes an action A in given state S. This might be the case when the agent is operating in the real world.



About; MDP



## Understanding further with code Implementation

- Introduce OpneAI gym
- Demonstrate the dual-taxi-v2 environment
  - State space size, Action space size
  - Transition table
  - Rendering
- Demonstrate the random agent
- Discuss two kinds of approaches
  - Cooperative
  - Competitive

- Random agents 



Introducing Q-learning

- Simplest Reinforcement Learning algorithm (Q-learning)

  - q-table significance

  - How are q-values updated?

    - Importance of different parameters
    - exploration-exploitation
    - discount factor
    - learning rate 


- Introduce the training and evaluation logic 
- Show some graphs
- Compare results, if possible show graphs side by side, and use seaborn for nice graphs? 



Common Challenges while applying Reinforcement Learning algorithms

- Stochasticity
- Partially observable environments
- State space and action space
  - Discrete v/s continuous
  - non convergence, local minimas
- Reward structure
  - Shapely value
  - Game theory (strictly dominant strategy, nash equilibrium)
- The algorithm fit
- Hyperparameter tuning

Questions

- Can I frame every problem as Reinforcement learning problem?
  - MDP

- Short note about various kinds of reinforcement learning algorithms 

## Further reading

- Richard Sutton
- YouTube lectures by David Silver
- Friend and foe Q-learning (Nash Equilibrium)
- blog series on medium by that guy



### Illustrations to make

- The state space table 
- The action space 
- Combine Q-table
- The analogy
- Plots



