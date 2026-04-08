#  RezzRankerV2  ( All-Mpnet-base-v2_ vanilla)
## The Upgrade to my last years V1 project, It has better logic, better accuracy, and better UI.

### It has been designed in phases with a pipeline= 
_________________________________________________
**Extract text (PyMuPDF + Tesseract → handles both normal PDFs and scanned ones)** 
_________________________________________________
**Preprocess (chunking + merging useful info for better semantic meaning)**
_________________________________________________
**Section extraction (skills, projects, experience, education)**
_________________________________________________
**Compare each section with JD using cosine similarity**
**Weighted scoring (skills > education etc.)**
_________________________________________________
**Final forward pass to combine everything efficiently**
_________________________________________________

Results-
SpearmanRank(old)=0.59 
SpearmanRank(New)=0.62 (looks small but in spearman terms thats significant) 81 % accuracy 

weblink(yes its streamlit but pretty)- https://lnkd.in/gyuFcEDF


Video Explanation of whole backend (YT)- https://lnkd.in/gKpNN-cT


Why its not perfect (Yet)- 

1) The model All-Mpnet-base-v2 (vanilla) is a general sentence embedding model and does not understand difference b/w domains ex- Can distinguish b/w tech and non tech resumes by itself... needs fine tuning.

 
2) Haven't Implemented skill importance weighting Ex- Python is treated same as time management skill.

3)The Chunking noise problem, the resume chunks despite alot of preprocessing is still not perfect and contain noise.




Thats V2 👇
(Took the most amount of work till yet)
