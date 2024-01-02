## Multi-Task Instruction Pretraining (MIP)

Multi-Task Instruction Pretraining (MIP) is  simple and effective when fine-tuning both specific-domain LLM and general LLM.

The central idea is concatenating the datasets from incremental pretraining(PT) and supervised fine-tuning(SFT) into one single dataset and training based on it.

T5 models, ExtrT5 models, and GLM-130B models all leverage multitask learning during training, and results indicate that multitask learning in pretraining is more beneficial than fine-tuning alone. 
Apart from large-scale models in the general domain, a vertical domain model, ChatHome, also highlights the performance of MIP, outperforming the 'PT+ SFT' paradigm.

