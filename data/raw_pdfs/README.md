# EAMCET PDFs Data Folder

Copy your EAMCET PDFs here in the following structure:

```
raw_pdfs/
├── EAMCET-AP/
│   ├── MPC/
│   │   ├── ap-eamcet-engineering-question-paper-2022-2034.pdf
│   │   ├── ap-eamcet-engineering-answer-keys-2024-2452.pdf
│   │   └── ...
│   └── BiPC/
│       ├── ap-eamcet-agriculture-question-paper-2023-2347.pdf
│       └── ...
└── EAMCET-TG/
    ├── MPC/
    └── BiPC/
```

The system will automatically:
- Detect state (AP/TG) from filename or folder structure
- Detect stream (MPC/BiPC) from filename keywords
- Identify paper type (question-paper, answer-keys, solutions)
- Extract year information from filename
