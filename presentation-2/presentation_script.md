# Presentation 2: Applying the model to a given Use-Case

## Guiding Questions:
- Explain how your approach could be used for our use case
  - keep it realistic
- Can your approach be used for this use case?
  - if not directly, why not?
  - Could you adapt your approach?
- How exactly would your approach work?
- What data would you need for your approach?
- How could you evaluate whether your approach works?

## What we get:
- Overall goal: Based on how **difficult** a specific learning objective is for a student, we want to **recommend them exercises fitting to their skill level**
- difficulty rating for each student and each learning objective (easy, medium, difficult)
- We don't get a difficulty rating for the exercises
- Example setting on Ilias

 
## Section 1: Recap
- Start with a short recap, which will summarize the most important concepts of HybridBERT4Rec

### An Introduction to Sequential Modelling
- Explain traditional CBF: 
  - Entire history is used, it is assumed that each item is equally important
  - This is insufficient modelling! → It can only model a users **<u>general</u>** preferences
  - But, user preferences change over time! → Users might have phases of a particular interest or preference... User A might generally like romantic films but recently got into the Lord of the Rings Franchise and wants to watch more LOTR movies / series → A traditional CBF recommender can't properly model this "temporary" spike of interest / relationship
- - Considers the **<u>order</u>** of users historical interactions → history is now modelled as a sequence
- This Sequential modelling of the history, implies that recent interactions are more relevant / important than the ones in the past, but at the same time covers past preferences as well
- It allows modelling "temporary" spikes of interest and "time capsule" relation → can recommend everything related to LOTR, covering the temporary interest, as well as providing recommendations that fit the users general interest in romantic films

### HybridBERT4Rec Architecture
- Consists of 3 parts:
  - CBF-BERT4Rec: Content Based Filtering Part
  - CF-BERT4Rec: Collaborative Filtering Part
  - Prediction: Layer
- CBF-BERT4Rec:
  - Takes the users history as a sequence as input
  - Applies a masked language modelling objective: a target item is masked during training which is then predicted, or during inference a mask token is appended to the sequence, to predict the next item a user will likely interact with
  - Data is then passed through BERT4Rec with a prediction head on top, which is the predecessor, on which HybridBERT4Rec is build upon
    - Recall: BERT4Rec is a standard BERT transformer that has been trained to perform content recommendation within a CBF setting
  - After training, this part will yield the distribution of all items over the target item, expressed as the interaction probability of all items with the target user, which the authors call the **target user $u$ profile $R_{uv}$**.
- CF-BERT4Rec:
  - Works in a similar fashion
  - Aim is to extract the target item representation based on the target user and its neighbors → target user profile
  - Input: The user sequence of the target item -> includes all users who have rated the target item including the target user → all users who have rated the item are assumed to be neighbors
  - then generates a distribution of all users over the target user -> we receive user-similarity probabilities between the neighboring users and the target user which the authors call the **target item $v$ profile $R_{vu}$**
- Prediction layer:
  - Uses a generalization of matrix factorization with neural networks
  - Combines the outputs of the previous two parts, in order to predict a rating the target user $u$ would assign to the target item $v$

## Section 2: The setting
- Users rate Learning objectives based on their difficulty for them
- We have:
  - A Collection of Users $U$
  - A Collection of Learning Objectives $T$
  - A Collection of Exercises $X$
- We can map each pair of Users $U$ and Learning Objectives $T$ to a difficulty rating $D$, where for all difficulty ratings $d$, holds that they are either 1,2 or 3 (easy, medium, difficult)
- We also have a constraint on our Exercise collection, where for each exercise $x$ in the Collection, there exists a learning objective $t$ from the Learning Objective Collection, such that a tuple of the exercise and learning objective can be build that is included in the Relation $R_{t,x}$, which maps each exercise to a Learning Objective

## Section 3: Model Adaptation

## Section 4: Evaluation

