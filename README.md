# Student Enquiries Classifier

This is an application designed to classify student enquiries into predefined classes, built on top of Natural Language Processing (NLP) technologies. The core of this project is built with the popular NLP library, SpaCy, and the MiniLM-L6-v21 Sentence Transformer.

## Authors and assignments

-   Luka Blašković (lblaskovi@student.unipu.hr)

## Short description of available functionalities
Users (students) can input their questions in Croatian language. The classifier will return a set of probabilities of those sentences belonging to predefined classes such as "final thesis" or "enrollment in second year". Also, the system will output an answer with highest similarity to the predefined set of answers. The model used is [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Tokenization and data embeddings are done using spaCy.  

### Technology Stack
- **SpaCy:** SpaCy is a free, open-source library for advanced NLP in Python. It's designed specifically for production use and helps us deal with large amounts of text data.

- **MiniLM-L6-v21 Sentence Transformer:** The MiniLM-L6-v21 Sentence Transformer is a transformer model optimized for generating sentence embeddings. This model is trained on a large corpus of sentences and can generate dense vector representations for sentences or paragraphs.

## Organization

[Juraj Dobrila University of Pula](http://www.unipu.hr/)  
[Pula Faculty of Informatics](https://fipu.unipu.hr/)  
Course: **Distributed Systems**, Academic Year 2022/2023  
Mentor: **doc. dr. sc. Nikola Tanković** (https://fipu.unipu.hr/fipu/nikola.tankovic, nikola.tankovic@unipu.hr)
