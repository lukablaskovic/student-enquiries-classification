# "Student Enquiries Classifier"

NLP application for classifying student enquiries in predefined classes.


### Authors and assignments

-   Luka Blašković (lblaskovi@student.unipu.hr)

### Short description of available functionalities
Users (students) can input their questions in Croatian language. The classifier will return a set of probabilities of those sentences belonging to predefined classes such as "final thesis" or "enrollment in second year". Also, the system will output an answer with highest similarity to the predefined set of answers. The model used is [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Tokenization and data embeddings are done using spaCy.  

### Organization

[Juraj Dobrila University of Pula](http://www.unipu.hr/)  
[Pula Faculty of Informatics](https://fipu.unipu.hr/)  
Course: **Distributed Systems**, Academic Year 2022./2023.  
Mentor: **Nikola Tanković** (https://fipu.unipu.hr/fipu/nikola.tankovic, nikola.tankovic@unipu.hr)
