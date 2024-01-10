
- Self-Conciusness module (First, we argue that simple task-agnostic pipelines for incontext learning should give way to deliberate, task-aware strategies.)
  
  - https://github.com/castillosebastian/dspy
  - Asses the quality of informations "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"
  - https://github.com/stanfordnlp/dspy
  - https://github.com/SqueezeAILab/LLMCompiler


# U4: Public Draft Filling Generator

1. **Query Expansion/Analyze**: The system expands (NER-extraction, hypothesis-answers, etc.) and analyzes the initial query to include related content.
2. **Retriever with Hybrid-Search-Filter/SQL**: Uses a hybrid search method + filter (when applicable) or SQL to gather relevant information.
3. **1-Generation with Self-Evaluation**: Generates an initial response from the LLM with self-evaluation on accuracy and relevance (prompt-programming).
4. **Retriever (Additional Information)**: Searches for additional information if the self-evaluation calls for more information.
5. **Augmentations**: Enriches the response with additional information (APIs and Agents).
6. **2-Generations**: Creates a second improved response based on 'consolidated' information (techniques of document-compression yet to be seen).
7. **Answer**: Presents a complete and detailed final response to the user.
8. Chat-history?
9. Demostrate-Search-Predict: python classes!
10. Self-Aware?
   
