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
  - This is insufficient modelling! -> It can only model a users **<u>general</u>** preferences
  - But, user preferences change over time! -> Users might have phases of a particular interest or preference... User A might generally like romantic films but recently got into the Lord of the Rings Franchise and wants to watch more LOTR movies / series -> A traditional CBF recommender can't properly model this "temporary" spike of interest / relationship

### Sequential modelling
- Considers the **<u>order</u>** of users historical interactions -> history is now modelled as a sequence
- This Sequential modelling of the history, implies that recent interactions are more relevant / important than the ones in the past, but at the same time covers past preferences as well
- It allows modelling "temporary" spikes of interest and "time capsule" relation -> can recommend everything related to LOTR, covering the temporary interest, as well as providing recommendations that fit the users general interest in romantic films

### Common Approaches to Sequential modelling
- Some variant of an RNN such as LSTMs or GRUs
- General idea: encode a user's historical records into a vector representation
- E.G. DREAM: Uses a representation of items (item IDs, latent features etc...) from a users' history to learn a representation of a user's interests at different times, which is then used to calculate a rating score for all items at each time step
- However, these Systems suffer from common RNN problems -> exploding and vanishing gradients, catastrophic forgetting, deep in time
- Consider Transformers!

## Section 2: HybridBERT4Rec

## Section 3: Performance

## Section 4: Applicability to E-Learning

## Section 5: Summary

## Discussion Topics / Additional slides: