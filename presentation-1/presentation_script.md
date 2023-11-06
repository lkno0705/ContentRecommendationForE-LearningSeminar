# Content Recommendation for E-Learning Seminar - Script Presentation 1
- **Title**: "HybridBERT4Rec: A Hybrid Recommender System Based on BERT"
- **Subtitle**: "Sequential Content-Based and Collaborative Filtering"
- **Presentation Date**: 08.11.23
- **Presentation Duration**: 15-20min
- **Discussion Duration**: ~10min

### Topics to be covered:
- How does your approach fit into the broader area of content recommendation (for e-learning)?
- How does your approach work?
- What are the strengths and weaknesses?
- What variants are there?
- Has it been used/evaluated in practice?
- How could this approach be extended?
- Prioritze and Structure what you want to say.

## Section 1: An Introduction to Sequential Content Recommendation

### The downsides of traditional CBF
- Explain traditional CBF: 
  - Entire history is used, it is assumed that each item is equally important
  - This is insufficient modelling! → It can only model a users **<u>general</u>** preferences
  - But, user preferences change over time! → Users might have phases of a particular interest or preference... User A might generally like romantic films but recently got into the Lord of the Rings Franchise and wants to watch more LOTR movies / series → A traditional CBF recommender can't properly model this "temporary" spike of interest / relationship

### Sequential modelling
- Considers the **<u>order</u>** of users historical interactions → history is now modelled as a sequence
- This Sequential modelling of the history, implies that recent interactions are more relevant / important than the ones in the past, but at the same time covers past preferences as well
- It allows modelling "temporary" spikes of interest and "time capsule" relation → can recommend everything related to LOTR, covering the temporary interest, as well as providing recommendations that fit the users general interest in romantic films

### Common Approaches to Sequential modelling
- Some variant of an RNN such as LSTMs or GRUs
- General idea: encode a user's historical records into a vector representation
- E.G. DREAM: Uses a representation of items (item IDs, latent features etc...) from a users' history to learn a representation of a user's interests at different times, which is then used to calculate a rating score for all items at each time step
- However, these Systems suffer from common RNN problems → exploding and vanishing gradients, catastrophic forgetting, deep in time, uni-directional
- Consider Transformers!

## Section 2: HybridBERT4Rec

### High Level Overview
- HybridBERT4Rec is a hybrid recommendation model, that uses Collaborative Filtering and Content-Based Filtering in order to predict a rating for a target item
- HybridBERT4Rec is composed of three main parts:
  - CBF-HybridBERT4Rec: Covers Content-Based Filtering
  - CF-HybridBERT4Rec: Covers Collaborative Filtering
  - Prediction Layer: Combines both outputs to predict a rating
- Before we can take a closer look at these we first have to understand its predecessor BERT4Rec which this model is based upon.

### BERT4Rec
- Task: Aims to predict the next item a user is likely to interact with
- Takes a sequence of items (tokens) from the users' history in **<u>chronological order</u>** as input.
- Randomly masks an item in the sequence during training, during inference the mask is appended to the input sequence
- Pass the masked sequence to a standard BERT transformer
- The resulting hidden representation of the masked item is then passed through a projection layer, which predicts the recommendation
-  → Essentially train with a Masked Language Modelling objective but with recommendation items → we predict the target item from an item context, which is given by a users' history → This plays directly into the strengths of the transformer architecture as they excel at building information rich contexts
- This allows the transformer to learn the relation between user' histories and thus current and past interests, and the target items
- A limitation of this approach, is that the set of recommended items for a given user will always be in one of the same categories the user has seen before → Because the prediction solely relies on the users history

### CF-HybridBERT4Rec: Collaborative Filtering
- Aim is to extract the target item representation based on the target user and its neighbors → target user profile
- Input: The user sequence of the target item -> includes all users who have rated the target item including the target user → all users who have rated the item are assumed to be neighbors
- The target user is masked in the input, during inference, a mask token is appended to the user sequence
- Use BERT4Rec as explained earlier
- After training we receive a model, which can predict an item representation that is composed of characteristics from all users that have rated the item → We essentially receive a representation about the target user profile for the target item → item representation based on user characteristics
- The projection layer then generates a distribution of all users over the target user -> we receive user-similarity probabilities between the neighboring users and the target user which the authors call the **target item $v$ profile $R_{vu}$**

### CBF-HybridBERT4Rec: Content Based Filtering
- Aim is to extract the target user representation that describes the target users preferences / interests
- Input: Sequence of items from the target users history, the target item is masked in the sequence, during inference a mask token is appended to the sequence
- Employs the same architecture / concepts as used in CF-HybridBERT4Rec but with different data
- Output in this case is the distribution of all items over the target item, expressed as the interaction probability of all items with the target user, which the authors call the **target user $u$ profile $R_{uv}$**.

### Prediction Layer: Combining CF & CBF
- Uses the target item profile $R_{vu}$ and target user profile $R_{uv}$ to predict the rating the target user $u$ would assign to the target item $v$
- It uses Generalized Matrix Factorization (GMF) which is a generalization of Matrix Factorization that uses neural networks with sigmoid activations.
- To be precise: $R(u,v) = R_{uv} \odot R_{vu}$, where $\odot$ denotes the element-wise product of the vectors, $\hat{r}_{u,v} = \sigma(WR(u,v) + b)$, where $\hat{r}_{u,v}$ denotes the predicted rating
- This approach allows the model to learn a variant of Matrix Factorization that isn't uniform, in other words, it can assign different weights to different latent dimensions of the user-item vector $R(u,v)$

### Strengths and Weaknesses of the architecture

- Strengths:
  - Highly parallelizable → Sequential in depth
  - Bi-directional → generates the context based on past and future information for each token in a sequence
  - Training is easy to parallelize → the CBF and CF model can be trained independent of each other, the prediction layer is then trained with frozen CBF and CF models
  - Inherits the strengths of sequential modeling for recommender systems
  - The CBF and CF model can be executed independent of each other → allowing for dynamic profile calculations whenever needed
  - Intermediate results can be cached → e.g. we only need to recalculate the target item profile if a new rating has been added to the target item, otherwise the old one can be reused.

- Weaknesses
  - Sequence length is limited because of memory constraints from the transformer models -> history is therefore also limited
  - Needs lots of Memory and Processing power in the worst case → We need to run a transformer model for each user and a separate model for each target item
  - It only works with rating information doesn't support implicit feedback such as clicking on items, etc.

## Section 3: Performance & Experiments

### Results
- show two plots with data
- Show that HybridBERT4Rec outperforms all models across the board, with the exception in the hit rate for low values of k
- Model has been evaluated on three datasets:
  - MovieLens
  - Yelp
  - Goodreads
- As Metrics the hit ratio (HR) of the top k predictions and normalized discounted cumulative gain (NDCG) were reported (for both higher is better)
- It was evaluated against 4 models:
  - Caser: unidirectional CBF approach using CNNs that predicts the top N-ranked items that a user will likely interact with
  - GRU4Rec: unidirectional CBF approach using RNNs that predicts next item embeddings
  - SAS4Rec: unidirectional CBF approach that uses a transformer-based architecture to predict the next item a user will interact with
  - BERT4Rec: as covered earlier, bidirectional CBF approach based on BERT

### The problems with the experiments
- NO HYBRID MODEL HAS BEEN EVALUATED!!!
  - The uplift in performance could be just because of the addition of the CF approach
  - The only thing we learned from the performance plots above is that BERT4Rec outperforms its competitors...
- Generalization performance is unknown as the authors didn't provide any information about how the data was partioned
  - Did they exclude users from the train set that were only in the test set? How does it handle unseen users and items → Does it suffer from the Cold start problem?
  - How does the model handle domain transfer after training?
  - Does it inherit the fine-tuning capabilities of transformers?
  - → real world applicability is also unknown

## Section 4: Applicability to E-Learning
- The model itself can easily be transferred to E-Learning as long as Timestamps, user and rating information is available for every item in the dataset.
- E.g. Imagine a linked-in learning setting, where for each training available the user ratings along with timestamps and a user-ID for each rating are available → Then the training and test cases used by the authors can be reconstructed by grouping the samples by user-ID and by training, yielding the required item and user sequences. The result would then be a ranking of the best trainings available in linked-ins database, that should be recommended to the user. With such information one could fill the landing page or a "For you page".
- Real world applicability is unknown, as mentioned before

<!-- ## Section 5: Summary -->

## Discussion Topics / Additional slides: