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
- We can map each pair of Users $U$ and Learning Objectives $T$ to a difficulty rating $D$, where each difficulty rating is either 1,2 or 3 (easy, medium, difficult)
- We also have a constraint on our Exercise collection, where for each exercise $x$ in the Collection, there exists a learning objective $t$ from the Learning Objective Collection, such that a tuple of the exercise and learning objective can be build that is included in the Relation $R_{t,x}$, which maps each exercise to a Learning Objective

## Section 3: Model Adaptation

### CBF-HybridBERT4Rec
- No architecture changes needed
- However, input data needs to be properly defined, as it is not given directly by the use case
- First lets look a bit closer at the history of a user H(u):
  - The history of user u is defined to be an ordered set of tuples $(x_i, t_j, s_k)$ for which hold, that the tuple $(x_i, t_j)$ is present in $R_{t,x}$, with the set being ordered by s, the timestamp, such that $s_{k-1} \leq s_k$ → The history of a user consists of the exercise and its corresponding topic the user has completed, along with the timestamp of the time of completion
- Based on that, we can define our model input $I(u)$ to be an ordered set of every $x_i$, which is included in the users' history $H(u)$. The set preserves / follows the same order as $H(u)$ → our model input at this point in time is the history of exercises of user $u$
- Model yields: the interaction probability distribution of all
items with the user u over the target item → we want to predict the item the user is most likely to interact with next based on its history
    - This prediction is not specific to a learning objective but is limited to the learning objectives present in the users' history
    - To push predictions further in the direction of a specific learning objective, we can filter for a specific $t$ when constructing I(u) → Increase compute costs significantly, as we'd need to run this CBF part $T\times U$ times.

### CF-HybridBERT4Rec
- No changes in terms of Architecture needed
- But the input data is created different
- We still input a sequence of users
- However, instead of considering all users who have rated the target learning objective (formerly the target item), we impose a more restrictive filtering constraint
- a user $u$ is in the set of Neighbors for target user $u_m$ and the target learning objective $t$ **if and only if**, the difficulty rating $d_{u,t}$ $u$ has assigned to t, is equal to the difficulty rating $d_{u_m, t}$, $u_m$ has assigned $t$ **and** if the target item $(x,t)$ is included in the users' history → Only users which gave the same difficulty rating to the current target learning objective as the target user and users' who already completed the target exercise are considered neighbors
- this model then yields, as before, a user-similarity probability distribution of all neighboring users over the target (masked) user
- By applying the aforementioned filtering criteria, we essentially limit the set of users who can receive a high similarity probability to users who have the same level of difficulty with the same learning objective → With that, we can later recommend exercises that may have helped others as well and these exercises are more likely to have a difficulty level that fits the target user.
- Without this filtering criteria, the target user may be deemed similar to another user who rated the target learning objective $t$ as easy, even though the target user has rated it as $difficult$. This could result in the recommendation of way to difficult exercises. → e.g. the user embeddings only differ very slightly except for the difficulty rating for topic t.
- This approach allows the recommendation of exercises from a completely different learning objective later on, if the user embeddings are not specific to the given learning objective.

### The complete model
- Explain final algorithm
- Yields a rating $\hat{r}_{u,x}$ for each exercise and for each user
- Construct an overall rating of exercises by sorting the ratings of all exercises regardless of their assigned learning objective
- Construct a topic specific rating by filtering for a learning objective and then sorting the ratings in each learning objective group

## Section 4: Evaluation

### Pooling
- The Problem: We have no test set that includes binary or graded relevance ratings and annotating the whole exercise collection for every user and every learning objective with those ratings is infeasible as this would require $U \times T \times X$ relevance annotations
- That's why we want to use a method called Pooling
- The idea:
  - For most queries (user, learning objective combinations) only a tiny fraction of exercises is actually relevant → only $N << X$ documents are actually relevant for our current query
  - An ideal retrieval system would of course rank these on top
  - → We only annotate the top $n$ results in a ranking → We only need $U \times T \times N$ annotations
- With these annotations, we can then compute the typical IR-Evaluation metrics, such as P@k, R@k, NDCG, AP, MAP etc.
- Shortcoming of this approach: it is not given that all relevant documents are included in the top $n$ positions of the ranking → we will ignore some relevant documents and thus the scores are only an approximation.
- Nevertheless, this method marks a great balance between annotation work and evaluation accuracy if configured well.
