
- Self-Conciusness module (First, we argue that simple task-agnostic pipelines for incontext learning should give way to deliberate, task-aware strategies.)
  
  - https://github.com/castillosebastian/dspy
  - Asses the quality of informations "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"
  - https://github.com/stanfordnlp/dspy
  - https://github.com/SqueezeAILab/LLMCompiler


#U4: Public Draft Filling Generator

1. **Query Expansion/Analyze**: El sistema amplía y analiza la consulta inicial para incluir términos y conceptos relacionados.
2. **Retriever with Hybrid-Search-Filter**: Utiliza un método de búsqueda híbrido para recopilar información relevante de múltiples fuentes.
3. **1-Generation with Self-Evaluation**: Genera una respuesta inicial y la evalúa para precisión y relevancia, ajustándola según sea necesario.
4. **Retriever (Additional Information)**: Busca información adicional si la autoevaluación indica la necesidad de más detalles.
5. **Augmentations**: Enriquece la respuesta con información adicional, ejemplos y mejores prácticas.
6. **2-Generations**: Crea una segunda respuesta mejorada basada en la información ampliada y revisada.
7. **Answer**: Presenta una respuesta final completa y detallada al usuario.
