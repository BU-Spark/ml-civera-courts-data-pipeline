# Technical Project Document Template

## Jeffrey Zhou, Christopher Ziko, Boyang Liu, Jamie Lin, 2024-Feb-15 v1.0.0

## Overview


The dataset is comprised of ~70,000,000 individual court actions. Given a short natural language description of the action, we are trying to label both the actor as well as the type of action (Case disposed, Motion allowed, Pre-trial Conference, etc.). "actor" refers to the main entities involved in the case, such as individuals or locations involved, while "action" denotes the specific types of behaviors or activities. The end goal is to be able to draw useful relations of court activity (eg. Frequency of a specific ruling by a specific judge given a specific type of case). Our task is to predict categories for missing entries.


%%% Our role within this is to provide the necessary labels.


### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.


The dataset is already partially labeled through a traditional algorithm mainly with regex, in cases where the labels can be directly deduced. We are trying to label the rest of the data with automation with AI. Given the large size of this dataset (~70,000,000), AI would be the most viable approach to this problem.


### B. Problem Statement:


For now, we consider this a classification problem to assign an actor and label based on each brief description of a case. We anticipate, however, that upon further analysis of the dataset, it is possible that we would have to generate additional labels. 


### C. Checklist for project completion


1. Consult with client and an source of legal expertise to create the set of labels
2. Train model capable of labeling the given dataset
3. Run the model on the dataset to fully label it


### D. Outline a path to operationalization.


The operational end goal is both a model that is capable of labeling future data, as well as the complete labeling of the client’s current data.


## Resources
1. Mass Court Plus (website created by Civera)
2. Probate Court Record


### Data Sets
* Data set is provided by the client
### References
None (for now)


## Weekly Meeting Updates


*Keep track of ongoing meetings in the Project Description document prepared by Spark staff for your project.*
Project manager will confirm client’s and team members’ schedule to set biweekly meetings
Possibly consult with experienced attorneys to determine proper labeling of court files 




Note: Once this markdown is finalized and merge, the contents of this should also be appended to the Project Description document.

